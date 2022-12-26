FileTxT = fopen('Mg_scale.txt');
format long
cell_data= textscan(FileTxT,'%s%s%s%s%s%s%s','Delimiter',' ');
CellData = cat(2,cell_data{:});
fclose(FileTxT);


Data = cellfun(@(x) extractAfter(x,":"),CellData(:,2:end),'UniformOutput',false);
Data = cellfun(@str2double,Data);
Data = Data.';

Label = cellfun(@str2double, CellData(:,1));

NumTrain = 1000;

TrainData = Data(:,1:NumTrain);
TrainLabel = Label(1:NumTrain);
TestData = Data(:,NumTrain+1:end);
TestLabel = Label(NumTrain+1:end);