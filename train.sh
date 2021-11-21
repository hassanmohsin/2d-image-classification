python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-one --model_dir models/70P.bce/point-one --batch_size 384 --epochs 50 --loss bce
python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-one --model_dir models/70P.focal/point-one --batch_size 384 --epochs 50 --loss focal

python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-two --model_dir models/70P.bce/point-two --batch_size 384 --epochs 50 --loss bce
python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-two --model_dir models/70P.focal/point-two --batch_size 384 --epochs 50 --loss focal

python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-three --model_dir models/70P.bce/point-three --batch_size 384 --epochs 50 --loss bce
python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-three --model_dir models/70P.focal/point-three --batch_size 384 --epochs 50 --loss focal

python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-four --model_dir models/70P.bce/point-four --batch_size 384 --epochs 50 --loss bce
python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-four --model_dir models/70P.focal/point-four --batch_size 384 --epochs 50 --loss focal

python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-five --model_dir models/70P.bce/point-five --batch_size 384 --epochs 50 --loss bce
python -m train.main --image_dir /home/mhassan/2d-dataset/70P/dataset/point-five --model_dir models/70P.focal/point-five --batch_size 384 --epochs 50 --loss focal
