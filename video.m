clear

camera = webcam;
load('detector.mat', 'detector');

while true
    picture = camera.snapshot;
    picture = imresize(picture, [32, 32]);
    
    [bboxes, scores] = detect(detector, picture);
    picture = insertObjectAnnotation(picture, 'rectangle', bboxes, scores);
    figure
    imshow(I)

%     label = classify(nnet, picture);
%     image(picture);
%     title(char(label));
%     drawnow;
end
