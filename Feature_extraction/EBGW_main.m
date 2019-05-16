clear all
clc
load train_data;
load test_data;
num1=numel(train_protein);
num2=numel(test_protein);
L=45;
for i=1:num1
eb1_train(i,:)= ebgw1(train_protein{i},L);
eb2_train(i,:)= ebgw2(train_protein{i},L);
eb3_train(i,:)= ebgw3(train_protein{i},L);
end
for i=1:num2
eb1_test(i,:)= ebgw1(test_protein{i},L);
eb2_test(i,:)= ebgw2(test_protein{i},L);
eb3_test(i,:)= ebgw3(test_protein{i},L);
end
ebgw_train=[eb1_train,eb2_train,eb3_train];
ebgw_test=[eb1_test,eb2_test,eb3_test];
save ebgw_45 ebgw_train ebgw_test

