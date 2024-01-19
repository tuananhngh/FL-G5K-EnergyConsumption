from jtop import jtop
import argparse
import logging
import os


def main():
    with jtop() as jetson:
        logging.info("Starting monitoring energy using jtop")
        result_file = args.log_dir+args.log_csv
        with open(result_file,"w+") as f:
            f.write(f"timestamp, RAM%, GPU%, GPU inst power, GPU avg power, CPU%, CPU inst power, CPU avg power, tot inst power, tot avg power\n")
            logging.info("Saving the power data in %s", result_file)
            while jetson.ok():
                stats = jetson.stats
                power = jetson.power
                cpu = sum([stats['CPU'+str(i)] for i in range(1,9)])/8
                line = f"{stats['time']}, {stats['RAM']}, {stats['GPU']}, {power['rail']['GPU']['power']/100}, {power['rail']['GPU']['avg']/100}, {cpu}, {power['rail']['CPU']['power']/100}, {power['rail']['CPU']['avg']/100},{power['tot']['power']/100}, {power['tot']['avg']/100}\n"
                f.write(line)
                f.flush()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', 
                        help='path to logs',
                        )
    parser.add_argument('--log-csv', 
                        help='csv file name',
                        default='energy.csv')
    args = parser.parse_args()

    logging.basicConfig(
        # filename=args.log_dir + args.log_file, 
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
        )
    
    try:
        main()
    except Exception as err:
        logging.error(err)