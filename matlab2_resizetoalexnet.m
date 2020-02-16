% load('myData.mat')
trainingData = myData.TrainData;
testData = myData.TestData;

% DataToResize = trainingData(1:200,:);
DataToResize = testData(1:100,:);

for n = 1 : height(DataToResize)
    % read each image
    I = imread(DataToResize.imageFilename{n});
    [height1, width1, ~] = size(I);

    % resize image to height 200, width resized automatically
    I2 = imresize(I,[224, 224]);
    [height2, width2, ~] = size(I2);
    
    % calculate resized difference
    resized_diff_height = height1 / height2;
    resized_diff_width = width1 / width2;
    
    % apply diff to bounding box
    bounding_box = DataToResize.vehicle{n};
    bounding_box(1) = bounding_box(1) / resized_diff_width;
    bounding_box(2) = bounding_box(2) / resized_diff_height;
    bounding_box(3) = bounding_box(3) / resized_diff_width;
    bounding_box(4) = bounding_box(4) / resized_diff_height;
    
    % round to the nearest integer.
    bounding_box = round(bounding_box, 0);
    
    % change 0 values to 1
    if(bounding_box(1) == 0)
        bounding_box(1) = 1;
    end
    if(bounding_box(2) == 0)
        bounding_box(2) = 1;
    end
    
    % write resized image
    resized_imagename = strcat('myData/', extractAfter(DataToResize.imageFilename{n}, "car_ims/"));
    imwrite(I2, resized_imagename);
    
    % insert resized bounding box to table
%     myResizedData.TrainData.imageFilename{n} = resized_imagename;
%     myResizedData.TrainData.vehicle{n} = bounding_box;
    
    myResizedData.TestData.imageFilename{n} = resized_imagename;
    myResizedData.TestData.vehicle{n} = bounding_box;
end
