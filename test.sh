#!/bin/sh

mode=$1
echo "mode: "${mode}

subworkdir=retinanet_x101_32x4d_fpn_1x_800x800_cls2
config="retinanet_x101_32x4d_fpn_1x.py"

echo "workdir: "${subworkdir}

work_dir="work_dirs/${subworkdir}/"

sudo chmod -R 777 ${work_dir}

#get results
python tools/test.py ./configs/hangkongbei/${config} \
     ${work_dir}"latest.pth" \
      --out ${work_dir}"results.pkl"


#evaluate
python ./tools/voc_eval.py ${work_dir}"results.pkl" "./configs/hangkongbei/${config}"
