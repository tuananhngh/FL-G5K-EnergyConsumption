from jtop import jtop
import argparse
import logging
import os


def main(args):
    with jtop() as jetson:
        logging.info("Starting monitoring energy using jtop.")
        result_file = os.path.join(args.log_dir,args.log_csv)
        with open(result_file,"w+") as f:
            f.write(f"timestamp,RAM%,GPU%,GPU inst power (mW),GPU avg power (mW),CPU%,CPU inst power (mW),CPU avg power (mW),tot inst power (mW),tot avg power (mW)\n")
            logging.info("Saving the power data in %s", result_file)
            while jetson.ok():
                stats = jetson.stats
                power = jetson.power
                cpu = sum([stats['CPU'+str(i)] for i in range(1,9)])/8
                line = f"{stats['time']}, {int(stats['RAM']*100)}, {int(stats['GPU']*100)}, {power['rail']['GPU']['power']}, {power['rail']['GPU']['avg']}, {int(cpu*100)}, {power['rail']['CPU']['power']}, {power['rail']['CPU']['avg']},{power['tot']['power']}, {power['tot']['avg']}\n"
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
        main(args)
    except Exception as err:
        logging.error(err)