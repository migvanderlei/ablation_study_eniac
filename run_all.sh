echo "Rodando todos os experimentos..."
python3.6 ablation.py --clf=0 --ds=books --task=1
python3.6 ablation.py --clf=1 --ds=books --task=1
python3.6 ablation.py --clf=0 --ds=tweets --task=1
python3.6 ablation.py --clf=1 --ds=tweets --task=1
python3.6 ablation.py --clf=0 --ds=restaurants --task=1
python3.6 ablation.py --clf=1 --ds=restaurants --task=1
python3.6 ablation.py --clf=0 --ds=books --task=2
python3.6 ablation.py --clf=1 --ds=books --task=2
python3.6 ablation.py --clf=0 --ds=tweets --task=2
python3.6 ablation.py --clf=1 --ds=tweets --task=2
python3.6 ablation.py --clf=0 --ds=restaurants --task=2
python3.6 ablation.py --clf=1 --ds=restaurants --task=2
python3.6 ablation.py --clf=0 --ds=apps --task=2
python3.6 ablation.py --clf=1 --ds=apps --task=2
echo "Experimentos finalizados"