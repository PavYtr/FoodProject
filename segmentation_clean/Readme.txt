``ImageSets'' folder, store the image ids of train and test sets of FoodSeg103:
        - train.txt
        - test.txt

``Images'' folder, contains the raw images and masks:
        - img_dir
                - train
                - test
        - ann_dir
                - train
                - test

category_id.txt: store the category ids and names

test_recipe1m_id.txt, train_test_recipe1m_id.txt: store the original image ids in Recipe1M dataset of FoodSeg103. You may explore more information based on these original image ids & recipe information in Recipe1M. 

(NOTE: There are some images in the list we do not use in the final FoodSeg103.)