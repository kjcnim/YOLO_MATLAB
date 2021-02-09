%% 사전 훈련된 검출기 다운로드 하기
doTraining = false;
if ~doTraining && ~exist('yolov2ResNet50VehicleExample_19b.mat','file')    
    disp('Downloading pretrained detector (98 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/yolov2ResNet50VehicleExample_19b.mat';
    websave('yolov2ResNet50VehicleExample_19b.mat',pretrainedURL);
end
%% 데이터셋 불러오기
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% Display first few rows of the data set.
vehicleDataset(1:4,:)

% Add the fullpath to the local vehicle data folder.
vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);

rng(0);
% shuffledIndices이 과정을 하는게 무슨 의미가 있나?
% height은 table의 행 수를 의미한다.
% 정수로 구성된 난수 순열로, randperm(n)이란 1에서 n까지의 정수가 반복되지 않게 구성된 난수 순열을 행 벡터로 반환한다.
% 따라서 shuffledIndices는 단지 데이터 개수 295개를 1부터 1간격으로 정수가 반복되지 않게 구성된 행 벡터이다. 
% 이를 하는 이유는 첫째 data개수를 알기 위함이다(?) 이를 통해 트레이닝 셋 등을 나눈다.
% 둘째, data set, training set, 검증 set으로 나눌때 사용되는 데이터를 무작위로 선별하기 위함이다.
% 그래서 밑에서 trainingDataTbl를 하면 shuffledIndices에서 나온 행렬의 있는 값에 해당되는 이미지 number가 적용함을 확인할 수 있다.
shuffledIndices = randperm(height(vehicleDataset));


%length(shuffledIndices) = 295
%idx = 177
idx = floor(0.6 * length(shuffledIndices) );

%trainingIdx = 1부터 차례로 177까지 행렬
trainingIdx = 1:idx;
%177개의 데이터(사진)과 그 값들을 불러옴. 이것이 훈련 세트
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

%178~ 207까지
validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

%208 ~ 295까지
testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

% 이거 왜하는 거지?
% imageDatastore와 boxLabelDatastore를 사용하여 훈련과 평과 과정에서 영상 및 레이블 데이터를 불러오기 위한
% 데이터 저장소를 만든다.
% imageDatastore : 이미지 데이터의 데이터저장소, imageFilename이란 제목의 행렬의 데이터를 불러온다.
% boxLabelDatastore : vehicle이란 제목의 행의 데이터를 가져온다.

imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

% 영상 데이터저장소와 상자 레이블 데이터저장소를 결합한다.
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

% 상자 레이블과 함께 훈련 영상 중 하나를 표시한다.
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% YOLO v2 객체 검출 신경망 만들기

% 예제를 실행하는데 소요되는 계산 비용을 줄이기 위해
% 신경망을 실행하는데 필요한 최소 크기인 [224 224 3]으로 신경망 입력 크기를 지정한다.
% 훈련 전 전처리 단계에서 영상의 크기를 조정하는 것이다.
inputSize = [224 224 3];

% 검출할 사물 클래스의 개수를 정의한다.
numClasses = width(vehicleDataset)-1;

% estimateAnchorBoxes를 사용하여 훈련 데이터의 사물 크기를 기반으로
% 앵커 상자를 추정한다.
% 훈련 전 이루어지는 영상 크기 조정을 고려하기 위해 앵커 상자 추정에 사용하는 훈련 데이터의 크기를 조정한다.
% transform을 사용하여 훈련 데이터를 전처리한 후에 앵커 상자의 개수를 정의하고 앵커 상자를 추정한다.
% preprocessData를 사용하여 훈련 데이터를 신경망의 입력 영상 크기로 크기 조정한다.
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

% 이제 resnet50을 사용하여 사전 훈련된 ResNet-50 모델을 불러온다.
featureExtractionNetwork = resnet50;

% activation_40_relu 뒤에 오는 layer들을 검출 하위 신경망으로 교체할 특징 추출 layer로 
% activation_40_relu를 선택한다.
% 이 정도의 다운샘플링은 공간 분해능과 추출된 특징의 강도 사이를 적절히 절충한 값이다.
% 신경망의 더 아래쪽에서 추출된 특징은 더 강한 영상 특징을 인코딩하나 공간 분해능이 줄어들기 때문이다.
% 최적의 특징 추출 계층을 선택하려면 경험적 분석이 필요하다.
featureLayer = 'activation_40_relu';

% YOLO v2 객체 검출 신경망을 만든다.
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

%% 데이터 증대

augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)

%% 훈련 데이터 전처리하기

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% YOLO v2 사물 검출기 훈련시키기
options = trainingOptions('sgdm', ...
        'MiniBatchSize',16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
    
if doTraining       
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options)
else
    % Load pretrained detector for the example.
    pretrained = load('yolov2ResNet50VehicleExample_19b.mat');
    detector = pretrained.detector;
end

I = imread(testDataTbl.imageFilename{1});
I = imresize(I,inputSize(1:2));

[bboxes,scores]  = detect(detector, I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

%% 테스트 세트를 사용하여 검출기 평가하기
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))





