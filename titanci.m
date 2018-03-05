clear;
data=load("train_modified.csv");
X = data(:, 2:7);
y = data(:, 1);
theta = learningAlg(X, y);
X_guess = load("test_modified.csv");
X_guess = mapFeature(X_guess);
h = sigmoid(X_guess * theta);
prediction = zeros(length(h),1);
prediction(find(h>=0.5)) = 1;

fprintf("Saving prediction to prediction.txt \n");
save prediction.txt prediction;