%Caricamento immagini di training
dataset_train = 'C:\Users\Lenovo\Desktop\a\Training'; 
folders_train = dir(dataset_train);
folders_train = folders_train([folders_train.isdir] & ~ismember({folders_train.name}, {'.', '..'}));

numClasses_train = length(folders_train);
featuresTrain = [];
labelsTrain = {};

for c = 1:numClasses_train
    classFolder_train = fullfile(dataset_train, folders_train(c).name);
    imageFiles_train = dir(fullfile(classFolder_train, '*.jpg'));
    
    for i = 1:length(imageFiles_train)
        imgPath_train = fullfile(classFolder_train, imageFiles_train(i).name);

        %Preprocessing
        img_train = im2double(imread(imgPath_train));
        grayImg_train = rgb2gray(img_train);
        enhancedImg_train = histeq(grayImg_train);
        grayImg_train = medfilt2(grayImg_train,[7 7]);
        
        %Segmentazione
        bwImg_train = sauvola(grayImg_train, [80 80]);
        bwImg_train = imerode(bwImg_train, strel("disk",6));
        bwImg_train = imclose(bwImg_train, strel("disk",10));
        bwImg_train = 1- bwImg_train;
        bwImg_train = imfill(bwImg_train, 'holes');
        bwImg_train = bwareaopen(bwImg_train, 70);
        bwImg_train = imclose(bwImg_train, strel("disk",40));
        bwImg_train = imfill(bwImg_train, 'holes');
        foglia_train = img_train.*bwImg_train;
        
        %Estrazione features
        %Descrittori Geometrici
        stats_train = regionprops(bwImg_train, 'Area', 'Perimeter', 'Eccentricity','Solidity','ConvexArea');
        area_train = stats_train.Area;
        perimeter_train = stats_train.Perimeter;
        eccentricity_train = stats_train.Eccentricity;
        solidity_train = stats_train.Solidity;
        convexarea_train = stats_train.ConvexArea;
        %Texture
        glcm_train = graycomatrix(enhancedImg_train, 'Offset', [0 1]);
        statsTexture_train = graycoprops(glcm_train, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
        %Colore
        meanColor_train = mean(reshape(foglia_train, [], 3));
        stdColor_train = std(double(reshape(foglia_train, [], 3)));

        featureVector_train = [area_train, perimeter_train, eccentricity_train, solidity_train,convexarea_train, ...
                         statsTexture_train.Contrast,statsTexture_train.Correlation,statsTexture_train.Energy, statsTexture_train.Homogeneity,...
                         meanColor_train, stdColor_train];

        if isempty(featuresTrain)
            featuresTrain = featureVector_train;
        else
            if length(featureVector_train) == size(featuresTrain, 2)
                featuresTrain = [featuresTrain; featureVector_train];
            end
        end

        %Salva l'etichetta solo se il vettore di feature è corretto
        if length(featureVector_train) == size(featuresTrain, 2)
            labelsTrain{end+1} = folders_train(c).name;
        end

    end
end

labelsTrain = categorical(labelsTrain);
disp(['Totale immagini di training elaborate: ', num2str(length(labelsTrain))]);


%Caricamento immagini di test
datasetPath_test = 'C:\Users\Lenovo\Desktop\a\Test'; % Modifica con il tuo percorso
folders_test = dir(datasetPath_test);
folders_test = folders_test([folders_test.isdir] & ~ismember({folders_test.name}, {'.', '..'}));

numClasses_test = length(folders_test);
featuresTest = [];
labelsTest = {};

for c = 1:numClasses_test
    classFolder_test = fullfile(datasetPath_test, folders_test(c).name); % Percorso della classe
    imageFiles_test = dir(fullfile(classFolder_test, '*.jpg')); % Trova file JPG
    
    for i = 1:length(imageFiles_test)
        imgPath_test = fullfile(classFolder_test, imageFiles_test(i).name);
        
        %Preprocessing
        img_test = im2double(imread(imgPath_test));
        grayImg_test = rgb2gray(img_test);
        enhancedImg_test = histeq(grayImg_test);
        grayImg_test = medfilt2(grayImg_test,[7 7]);
        
        %Segmentazione
        bwImg_test = sauvola(grayImg_test, [80 80]);
        bwImg_test = imerode(bwImg_test, strel("disk",6));
        bwImg_test = imclose(bwImg_test, strel("disk",10));
        bwImg_test = 1- bwImg_test;
        bwImg_test = imfill(bwImg_test, 'holes');
        bwImg_test = bwareaopen(bwImg_test, 70);
        bwImg_test = imclose(bwImg_test, strel("disk",80));
        bwImg_test = imfill(bwImg_test, 'holes');
        foglia_test = img_test.*bwImg_test;
        %Estrazione features
        %Descrittori geometrici
        stats_test = regionprops(bwImg_test, 'Area', 'Perimeter', 'Eccentricity','Solidity','ConvexArea');
        area_test = stats_test.Area;
        perimeter_test = stats_test.Perimeter;
        eccentricity_test = stats_test.Eccentricity;
        solidity_test = stats_test.Solidity;
        convexarea_test = stats_test.ConvexArea;
        %Texture
        glcm_test = graycomatrix(enhancedImg_test, 'Offset', [0 1]);
        statsTexture_test = graycoprops(glcm_test, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
        %Colore
        meanColor_test = mean(reshape(foglia_test, [], 3));
        stdColor_test = std(double(reshape(foglia_test, [], 3)));

        featureVector_test = [area_test, perimeter_test, eccentricity_test,solidity_test,convexarea_test, ...
                        statsTexture_test.Contrast,statsTexture_test.Correlation,statsTexture_test.Energy, statsTexture_test.Homogeneity,...                        
                        meanColor_test, stdColor_test];

        if isempty(featuresTest)
            featuresTest = featureVector_test;
        else
            if length(featureVector_test) == size(featuresTest, 2)
                featuresTest = [featuresTest; featureVector_test];
            end
        end

        if length(featureVector_test) == size(featuresTest, 2)
            labelsTest{end+1} = folders_test(c).name;
        end

    end
end


labelsTest = categorical(labelsTest);

disp(['Totale immagini di test elaborate: ', num2str(length(labelsTest))]);

%Addestramento del modello sul training set
model = fitcknn(featuresTrain, labelsTrain,'NumNeighbors',1);

save('modKNN.mat','model');

%Confusion matrix sul training set
pred_train = predict(model, featuresTrain);
cm_train = confmat(labelsTrain,pred_train);
figure(1),
show_confmat(cm_train.cm_raw,cm_train.labels);

%TEST
%Confusion matrix sul test set
pred_test = predict(model, featuresTest);
cm_test = confmat(labelsTest,pred_test);
figure(2),
show_confmat(cm_test.cm_raw,cm_test.labels);


 



