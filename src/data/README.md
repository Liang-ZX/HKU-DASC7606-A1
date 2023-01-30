# Dataset README

## Dataset Description

This dataset is a 2D image dataset for object detection in autonomous driving, which is created by HUAWEI Company.
The dataset contains 5K labeled training images, 2.5K labeled val images, and 2.5K images with hidden annotations.

There are only 5 categories in this dataset.
```python
CAR_CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram']
```

## The Dataset Organization
```python
- annotations
    - instance_train.json # annotation for train data
    - instance_val.json   # annotation for val data
    - instance_test.json  # annotation for test data. It will be used to evaluate your model outputs and not be released
- train
    - image
- val
    - image
- test   # It will be released later.
    - image
```

## The Annotation Format
```shell
"annotations": {
    "image_name": <str>  # The image name for this annotation.
    "category_id": <int>  # The category id.
    "bbox": <list>  # Coordinate of boundingbox [x, y, w, h].
}

"categories": {
    "name": <str>  # Unique category name.
    "id": <int>   # Unique category id.
    "supercategory": <str>  # The supercategory for this category.
}
```
