# Cupy-Covert-Channel


## TRAINING MODEL

    python training.py --split 10 --data dataset/sort_transpose/ --op transpose sort


## SENDER OF SECRET MESSAGE of COVERT CHANNEL

    python sender.py --op transpose sort --model model/\[\'transpose\'\,\ \'sort\'\]_1.0000.joblib --message 1010




## RECEIVER OF SECRET MESSAGE

    python receiver.py --folder message --op transpose sort --model model/\[\'transpose\'\,\ \'sort\'\]_1.0000.joblib --test 1010

