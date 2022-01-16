set -xe

data_dir="../dataset/Taobao"

./uicpr -train_ui ../dataset/Taobao/raw_sample.csv.pos.tsv \
        -train_ui_neg ../dataset/Taobao/raw_sample.csv.neg.tsv \
        -train_up ../dataset/Taobao/user_profile.csv.age_level.tsv,../dataset/Taobao/user_profile.csv.cms_group_id.tsv,../dataset/Taobao/user_profile.csv.cms_segid.tsv,../dataset/Taobao/user_profile.csv.final_gender_code.tsv,../dataset/Taobao/user_profile.csv.new_user_class_level.tsv,../dataset/Taobao/user_profile.csv.occupation.tsv,../dataset/Taobao/user_profile.csv.pvalue_level.tsv,../dataset/Taobao/user_profile.csv.shopping_level.tsv\
        -train_im ../dataset/Taobao/ad_feature.csv.adgroup_id.tsv,../dataset/Taobao/ad_feature.csv.brand.tsv,../dataset/Taobao/ad_feature.csv.campaign_id.tsv,../dataset/Taobao/ad_feature.csv.cate_id.tsv,../dataset/Taobao/ad_feature.csv.customer.tsv,../dataset/Taobao/ad_feature.csv.price.tsv \ \
        -update_times 1000 -dimension 64 -worker 10


# #
#,../dataset/Taobao/ad_feature.csv.brand.tsv,../dataset/Taobao/ad_feature.csv.campaign_id.tsv,../dataset/Taobao/ad_feature.csv.cate_id.tsv,../dataset/Taobao/ad_feature.csv.customer.tsv,../dataset/Taobao/ad_feature.csv.price.tsv \
# -train_ui_neg ../dataset/Taobao/raw_sample.csv.neg.tsv \