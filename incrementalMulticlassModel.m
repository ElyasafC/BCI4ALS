classdef incrementalMulticlassModel
    %INCREMENTALMULTICLASSMODEL class for multiclass incremental learning
    %  holds 3 models
    
    properties
        incMdl1
        incMdl2
        incMdl3
    end
    
    methods
        function obj = incrementalMulticlassModel(Training,LabelTraining)
            %INCREMENTALMULTICLASSMODEL Construct an instance of this class
            assert(all(LabelTraining<= 3))
            assert(all(LabelTraining >= 1))
            label1 = LabelTraining == 1;
            label2 = LabelTraining == 2;
            label3 = LabelTraining == 3;
            Mdl1 = fitcsvm(Training,label1);
            Mdl2 = fitcsvm(Training,label2);
            Mdl3 = fitcsvm(Training,label3);
            obj.incMdl1 = incrementalLearner(Mdl1);
            obj.incMdl2 = incrementalLearner(Mdl2);
            obj.incMdl3 = incrementalLearner(Mdl3);
        end
        
        function [negloss_combine, yPreds] = predict(obj, testSet)
            % input: testset : n * feature_length
            %output: negloss_combine: negative loss per class :  n * 3
            %        yPreds: chosen class : n
            [yPred1, negloss1] = predict(obj.incMdl1, testSet);
            [yPred2, negloss2] = predict(obj.incMdl2, testSet);
            [yPred3, negloss3] = predict(obj.incMdl3, testSet);
            negloss_combine = [negloss1(:,2), negloss2(:,2), negloss3(:,2)]; % might need the negloss1(:,2)
            [maxScores, yPreds] =  max(negloss_combine,[], 2);
        end
        
        function valAccuracy = calcAccuracy(obj, testSet, labels)
            % returns accuracy average on testset (scalar) 
            [maxScores, yPreds] = obj.predict(testSet);
            valAccuracy = sum(yPreds == labels)/length(labels);
        end
        
        function obj = OnlineLearn(obj, EEG_Features,currentClass)
            % recieves examples (EEG_Features) and tags (currentClass)
            % trains 
            label1 = currentClass == 1;
            label2 = currentClass == 2;
            label3 = currentClass == 3;
            obj.incMdl1 = updateMetricsAndFit(obj.incMdl1,EEG_Features,label1);
            obj.incMdl2 = updateMetricsAndFit(obj.incMdl2,EEG_Features,label2);
            obj.incMdl3 = updateMetricsAndFit(obj.incMdl3,EEG_Features,label3);
        end
        
        function save(obj, dest)
            if nargin == 1
                dest = 'trainedModel.mat'; %dest is optional parameter, this is default value
            end
            trainedModel = obj;
            save(dest, 'trainedModel')
            disp(strcat('saved model to ', dest)) 
        end
        
    end
end



