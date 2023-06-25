close all;
clear;
clc;


mask_path = 'F:\Night_S1\dataset5\train\mask\';
t1_path = 'F:\Night_S1\dataset5\train\t1ori\';
t2_path = 'F:\Night_S1\dataset5\train\t2ori\';
t3_path = 'F:\Night_S1\dataset5\train\t3ori\';
out_path = 'F:\Night_S1\dataset5\train\';

if exist(strcat(out_path,'t3'),'dir')
    rmdir(strcat(out_path,'t3'),'s');
end
mkdir(strcat(out_path,'t3'));

if exist(strcat(out_path,'t2'),'dir')
    rmdir(strcat(out_path,'t2'),'s');
end
mkdir(strcat(out_path,'t2'));

if exist(strcat(out_path,'t1'),'dir')
    rmdir(strcat(out_path,'t1'),'s');
end
mkdir(strcat(out_path,'t1'));

if exist(strcat(out_path,'label'),'dir')
    rmdir(strcat(out_path,'label'),'s');
end
mkdir(strcat(out_path,'label'));

t1out = strcat(out_path,'t1\');
t2out = strcat(out_path,'t2\');
t3out = strcat(out_path,'t3\');
labelout = strcat(out_path,'label\');

mask_list = dir(strcat(mask_path,'*.tif'));
t1_list = dir(strcat(t1_path,'*.tif'));
t2_list = dir(strcat(t2_path,'*.tif'));
t3_list = dir(strcat(t3_path,'*.tif'));

masksnum = length(mask_list);
for i = 1:length(t1_list)
    t1 = imread(strcat(t1_path,t1_list(i).name));
    t2 = imread(strcat(t2_path,t2_list(i).name));
    t3 = imread(strcat(t3_path,t3_list(i).name));
    
    for j = 1:masksnum
        mask = imread(strcat(mask_path,mask_list(j).name));
        mask = mask./255;
        t1_ = t1;
        t2_ = t2.*mask;
        t3_ = t3;
        
        str1 = strcat(t1out,mask_list(j).name(1:end-4),'_',t1_list(i).name);
        str2 = strcat(t2out,mask_list(j).name(1:end-4),'_',t2_list(i).name);
        str3 = strcat(t3out,mask_list(j).name(1:end-4),'_',t3_list(i).name);
        str4 = strcat(labelout,mask_list(j).name(1:end-4),'_',t2_list(i).name);
        imwrite2tif(t1_,[],str1,'single','Copyright','MRI', 'Compression',1);
        imwrite2tif(t2_,[],str2,'single','Copyright','MRI', 'Compression',1);
        imwrite2tif(t3_,[],str3,'single','Copyright','MRI', 'Compression',1);
        imwrite2tif(t2,[],str4,'single','Copyright','MRI', 'Compression',1);
        disp(str2);
    end

end