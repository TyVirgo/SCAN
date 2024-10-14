# main
python main.py --model scan --data_test Set5 --data_range 801-900 --scale 2 --save rfdn_x4 --pre_train ../scan_x2.pt --rgb_range 1 --test_only --save_results
python main.py --model scan --data_test Set5 --data_range 801-900 --scale 3 --save rfdn_x4 --pre_train ../scan_x3.pt --rgb_range 1 --test_only --save_results
python main.py --model scan --data_test Set5 --data_range 801-900 --scale 4 --save rfdn_x4 --pre_train ../scan_x4.pt --rgb_range 1 --test_only --save_results
