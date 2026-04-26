_base_ = './faster_rcnn_r50_rgb.py'

data = dict(
    test=dict(
        ann_file='data/test_rgb/annotations_RGB.json',
        img_prefix='data/test_rgb/'
    )
)
