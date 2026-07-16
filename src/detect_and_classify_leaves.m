% Pipeline di localizzazione, estrazione feature e classificazione multi-foglia.

clear; clc; close all;

%% Caricamento del Modello Addestrato
% Carica il modello KNN salvato in precedenza
if exist('modKNN.mat', 'file')
    load('modKNN.mat'); % Carica la variabile 'model'
else
    error('File modKNN.mat non trovato. Assicurati di averlo inserito nella directory corrente o nel path.');
end

%% Caricamento Immagine di Input (Collage)
imgName = "foglie.jpg"; % Modifica con il nome effettivo del tuo collage (es. "Copia di Aggiungi corpo del testo.jpg")
if ~exist(imgName, 'file')
    error('Immagine %s non trovata.', imgName);
end

img = im2double(imread(imgName));

%% Preprocessing (Scala di grigi + Equalizzazione + Filtro Mediano)
grayImg = rgb2gray(img);
enhancedImg = histeq(grayImg);
grayImgFiltered = medfilt2(grayImg, [7 7]);

%% Segmentazione delle Foglie (Sauvola + Morfologia Matematica)
bwImg = sauvola(grayImgFiltered, [80 80]);
bwImg = imerode(bwImg, strel("disk", 10));
bwImg = 1 - bwImg; % Inversione (foglie bianche su sfondo nero)
bwImg = imfill(bwImg, 'holes'); % Riempimento buchi interni
bwImg = bwareaopen(bwImg, 70);   % Rimozione del rumore a piccoli pixel
bwImg = imclose(bwImg, strel("disk", 8));
bwImg = imerode(bwImg, strel("disk", 2));
bwImg = imfill(bwImg, 'holes');

% Applica la maschera all'immagine originale per mantenere solo le foglie colorate
segmentedImg = img .* bwImg;

%% Identificazione delle Regioni Connesse (Foglie Singole)
% Estraiamo i descrittori geometrici locali direttamente con regionprops
props = regionprops(bwImg, 'BoundingBox', 'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'ConvexArea');

%% Visualizzazione Risultati e Loop di Classificazione
figure('Name', 'Classificazione delle foglie con contorni e etichette', 'NumberTitle', 'off');
imshow(img);
hold on;

numDetected = length(props);
fprintf('Rilevate %d foglie nel collage.\n', numDetected);

for i = 1:numDetected
    % Bounding box della foglia corrente
    box = props(i).BoundingBox;
    x = round(box(1)); 
    y = round(box(2));
    w = round(box(3)); 
    h = round(box(4));
    
    % Evita ritagli fuori dai limiti dell'immagine
    x = max(1, x); y = max(1, y);
    w = min(size(img, 2) - x, w);
    h = min(size(img, 1) - y, h);
    
    % Ritaglio della regione locale della foglia
    leafRegion = imcrop(segmentedImg, [x y w h]);
    leafRegionGray = imcrop(enhancedImg, [x y w h]); % Per calcolare la texture GLCM localmente
    
    % --- Estrazione Features Geometriche Locali ---
    area = props(i).Area;
    perimeter = props(i).Perimeter;
    eccentricity = props(i).Eccentricity;
    solidity = props(i).Solidity;
    convexarea = props(i).ConvexArea;
    
    % --- Estrazione Feature di Texture Locale (GLCM) ---
    % Nota: se l'immagine ritagliata è troppo piccola, graycomatrix potrebbe fallire
    if size(leafRegionGray, 1) > 1 && size(leafRegionGray, 2) > 1
        glcm = graycomatrix(leafRegionGray, 'Offset', [0 1]);
        statsTexture = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
        contrast = statsTexture.Contrast;
        correlation = statsTexture.Correlation;
        energy = statsTexture.Energy;
        homogeneity = statsTexture.Homogeneity;
    else
        contrast = 0; correlation = 0; energy = 0; homogeneity = 0;
    end
    
    % --- Estrazione Feature di Colore Locale ---
    % Calcoliamo la media e la deviazione standard escludendo lo sfondo nero del ritaglio
    maskCrop = imcrop(bwImg, [x y w h]);
    pixelsR = leafRegion(:, :, 1);
    pixelsG = leafRegion(:, :, 2);
    pixelsB = leafRegion(:, :, 3);
    
    % Considera solo i pixel appartenenti alla foglia (valore di maschera > 0)
    leafPixelsR = pixelsR(maskCrop > 0);
    leafPixelsG = pixelsG(maskCrop > 0);
    leafPixelsB = pixelsB(maskCrop > 0);
    
    if ~isempty(leafPixelsR)
        meanColor = [mean(leafPixelsR), mean(leafPixelsG), mean(leafPixelsB)];
        stdColor = [std(double(leafPixelsR)), std(double(leafPixelsG)), std(double(leafPixelsB))];
    else
        meanColor = [0, 0, 0];
        stdColor = [0, 0, 0];
    end
    
    % --- Creazione Vettore Feature (15-D) ---
    featureVector = [area, perimeter, eccentricity, solidity, convexarea, ...
                     contrast, correlation, energy, homogeneity, ...
                     meanColor, stdColor];
                 
    % --- Predizione con il Modello KNN ---
    try
        predictedLabel = predict(model, featureVector);
        % Se la label restituita è di tipo categorical, convertila in stringa
        if iscategorical(predictedLabel)
            predictedLabelStr = char(predictedLabel);
        else
            predictedLabelStr = predictedLabel;
        end
    catch ME
        warning('Errore nella classificazione della foglia %d: %s', i, ME.message);
        predictedLabelStr = 'Unknown';
    end
    
    % --- Disegno Grafico del Bounding Box ---
    rectangle('Position', box, 'EdgeColor', 'b', 'LineWidth', 2);
    
    % Posiziona il testo centrato o sopra il box
    text(x, y - 10, predictedLabelStr, 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
end

hold off;
title('Classificazione finale delle foglie con Bounding Box ed Etichette');