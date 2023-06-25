close all;
clear;
clc;
% T1 T2 T3Data pair
% If the T2 missing area is not completely complete, the mean value of T1 and T3 will be added. T1 and T3 missing areas are treated with nearest interpolation
% Generated after t1 t2 t3 data, and then to mask processing.
folder = 'F:\Day_S0\dataset2\Day_S4_fanzhuan30\';%Cropped image directory for all dates. This directory is the date directory, and there are multiple images of the same date in the date directory
out_path = 'F:\Day_S0\dataset2\train\';%Output a path where the t1 t2 t3 directory is automatically generated

date_list = dir(folder);
date_list(1)=[];date_list(1)=[];

% Maximum and minimum values
maxvalue = 3.68;
minvalue = -0.75;
%Create folder
if exist(strcat(out_path,'t1ori'),'dir')
    rmdir(strcat(out_path,'t1ori'),'s');
end
mkdir(strcat(out_path,'t1ori'));

if exist(strcat(out_path,'t2ori'),'dir')
    rmdir(strcat(out_path,'t2ori'),'s');
end
mkdir(strcat(out_path,'t2ori'));

if exist(strcat(out_path,'t3ori'),'dir')
    rmdir(strcat(out_path,'t3ori'),'s');
end
mkdir(strcat(out_path,'t3ori'));
t1_outpath = strcat(out_path,'t1ori\');
t2_outpath = strcat(out_path,'t2ori\');
t3_outpath = strcat(out_path,'t3ori\');
% The first layer traverses all dates
for k = 1:length(date_list)-2
    date_name = date_list(k).name;
    t1_name = date_name;
    [t2_name,t1h] = time_add(t1_name,1);
    [t3_name,t2h] = time_add(t2_name,1);
    
    t1_imgs_path = strcat(folder,t1_name,'\');
    t2_imgs_path = strcat(folder,t2_name,'\');
    t3_imgs_path = strcat(folder,t3_name,'\');
    
    t1_imgs_list = dir(strcat(t1_imgs_path,'*.tif'));
    t2_imgs_list = dir(strcat(t2_imgs_path,'*.tif'));
    t3_imgs_list = dir(strcat(t3_imgs_path,'*.tif'));
    
    num = 0;
    num2 = 1;
    % Date all present
    if (t1h>=1)&&(t1h<=31)&&(t2h>=1)&&(t2h<=31)
        for i=1:length(t1_imgs_list)
            % 3 image pairs to be cropped
            t1_img = double(imread(strcat(t1_imgs_path,t1_imgs_list(i).name)));
            t2_img = double(imread(strcat(t2_imgs_path,t2_imgs_list(i).name)));
            t3_img = double(imread(strcat(t3_imgs_path,t3_imgs_list(i).name)));
            [h,w] = size(t1_img);
            img_k = 1;%Mark cropped image block
            for x = 1:4:h-23
                for y = 1:4:w-23
                    % Conditions are met: All three images are complete data
                    tmp1_img = t1_img(x:x+23, y:y+23);
                    tmp2_img = t2_img(x:x+23, y:y+23);
                    tmp3_img = t3_img(x:x+23, y:y+23);
                    % Image blocks in the same position must have all pixels greater than 0
                    cond1 = all(all(tmp1_img>0));
                    cond2 = all(all(tmp2_img>0));
                    cond3 = all(all(tmp3_img>0));
                    % Image blocks in the same position, requiring 90% pixels greater than 0, and then interpolated. It is required to be within the China region, that is, the -1 value is not included
                    % Do outlier processing, check whether there is an outlier after interpolation, if there is an outlier, it is not used as the sample set
                    % 24*24*0.9 = 518
                    % 24*24*0.85=490
                    cond4 = sum(sum(tmp1_img>0))>547 & sum(sum(tmp1_img==-1))<5;
                    cond5 = sum(sum(tmp2_img>0))>547 & sum(sum(tmp2_img==-1))<5;
                    cond6 = sum(sum(tmp3_img>0))>547 & sum(sum(tmp3_img==-1))<5;
                    img_k = img_k+1;
                    if cond1&&cond2&&cond3
                        % output path
                        str1 = strcat(t1_outpath,t1_imgs_list(i).name(1:end-4),'_',num2str(img_k),'.tif');
                        str2 = strcat(t2_outpath,t2_imgs_list(i).name(1:end-4),'_',num2str(img_k),'.tif');
                        str3 = strcat(t3_outpath,t3_imgs_list(i).name(1:end-4),'_',num2str(img_k),'.tif');
                        imwrite2tif(tmp1_img,[],str1,'single','Copyright','MRI', 'Compression',1);
                        imwrite2tif(tmp2_img,[],str2,'single','Copyright','MRI', 'Compression',1);
                        imwrite2tif(tmp3_img,[],str3,'single','Copyright','MRI', 'Compression',1);
                        num = num+1;
                   
                    elseif cond4&&cond5&&cond6
                        %output path
                        tmp1_img(tmp1_img == 0) = nan;%Convert 0 values to nan values
                        tmp1_img(tmp1_img == -1) = nan;%Convert -1 values to nan values
                        tmp3_img(tmp3_img == 0) = nan;%Convert 0 values to nan values
                        tmp3_img(tmp3_img == -1) = nan;%Convert -1 values to nan values
                        s = size(tmp3_img);
                        
                        valid = find(~isnan(tmp1_img));
                        [l,p] = ind2sub(s,valid);%Converts NaN values to subscripts
                        v = tmp1_img(valid);
                        [ii,jj] = ndgrid(1:s(1),1:s(2));% Create a grid of the same size
                        tmp1_img = griddata(l,p,v,ii,jj,'nearest');
                        
                        
                        valid = find(~isnan(tmp3_img));
                        [l,p] = ind2sub(s,valid);%Converts NaN values to subscripts
                        v = tmp3_img(valid);
                        [ii,jj] = ndgrid(1:s(1),1:s(2));% Create a grid of the same size
                        tmp3_img = griddata(l,p,v,ii,jj,'nearest');
                        
                        % Make a mask for T2
                        t2_mask = tmp2_img;
                        t2_mask(t2_mask>0)=3;
                        t2_mask(t2_mask==-1)=1;
                        t2_mask(t2_mask==0)=1;
                        t2_mask(t2_mask==3)=0;
                        
                        tmp2_img(tmp2_img==-1)=0;
                        tmp2_img = tmp2_img+(tmp1_img+tmp3_img)./2.*t2_mask;
                        
                        % Maximum and minimum values
                        if max(max(tmp1_img))<maxvalue && min(min(tmp1_img))>minvalue &&...
                                max(max(tmp2_img))<maxvalue && min(min(tmp2_img))>minvalue &&...
                                max(max(tmp3_img))<maxvalue && min(min(tmp3_img))>minvalue
                            
                            str1 = strcat(t1_outpath,t1_imgs_list(i).name(1:end-4),'_c',num2str(img_k),'.tif');
                            str2 = strcat(t2_outpath,t2_imgs_list(i).name(1:end-4),'_c',num2str(img_k),'.tif');
                            str3 = strcat(t3_outpath,t3_imgs_list(i).name(1:end-4),'_c',num2str(img_k),'.tif');
                            imwrite2tif(tmp1_img,[],str1,'single','Copyright','MRI', 'Compression',1);
                            imwrite2tif(tmp2_img,[],str2,'single','Copyright','MRI', 'Compression',1);
                            imwrite2tif(tmp3_img,[],str3,'single','Copyright','MRI', 'Compression',1);
                            num = num+1;
                        else
                            disp(['outlier',num2str(num2),str2]);
                            num2 = num2+1;
                        end
                        
                    end
                end
            end
        end
        disp(['data:',t2_imgs_path,'amount:',num2str(num)]); 
    end    
end