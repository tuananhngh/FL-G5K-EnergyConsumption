from jtop import jtop
import argparse
import logging
import os

def get_process_headline():
    line = f"\
timestamp,\
process_name,\
process_id,\
cpu_util(%),\
mem,\
GPU_mem,\
state\
\n"
    return line
                

def get_process_line(stats, process):
    process_info = f"\
{stats['time']},\
{process[9]},\
{process[0]},\
{process[6]},\
{process[7]},\
{process[8]},\
\n"
    return process_info

def get_stats_headline():
    line = f"\
timestamp,\
RAM%,\
GPU%,\
GPU inst power (mW)\
,GPU avg power (mW)\
,CPU%\
,CPU inst power (mW)\
,CPU avg power (mW)\
,tot inst power (mW)\
,tot avg power (mW)\
\n"
    return line

def get_stats_line(stats, power):
    cpu = sum([stats['CPU'+str(i)] for i in range(1,9)])/8
    line = f"\
{stats['time']},\
{int(stats['RAM']*100)},\
{int(stats['GPU']*100)},\
{power['rail']['GPU']['power']},\
{power['rail']['GPU']['avg']},\
{int(cpu*100)},\
{power['rail']['CPU']['power']},\
{power['rail']['CPU']['avg']},\
{power['tot']['power']}, \
{power['tot']['avg']}\
\n"
    return line
                    

def get_jetson_info(jetson):
    stats = jetson.stats
    power = jetson.power
    processes = jetson.processes
    return stats, power, processes

def main(args):
    with jtop() as jetson:
        logging.info("Starting monitoring energy using jtop.")
        energy_file_path = os.path.join(args.log_dir,args.energy_csv)
        with open(energy_file_path,"w+") as energy_file:
            headline = get_stats_headline()
            energy_file.write(headline)
            logging.info("Saving the power data in %s", energy_file_path)
            
            process_file_path = os.path.join(args.log_dir,args.process_csv)
            with open(process_file_path,"w+") as process_file:
                headline = get_process_headline()
                process_file.write(headline)
                logging.info("Saving the process data in %s", process_file_path)
            
                while jetson.ok():
                    stats, power, processes = get_jetson_info(jetson)
                    # update process file
                    for process in processes:
                        line = get_process_line(stats, process)
                        process_file.write(line)
                        process_file.flush()
                    # update energy file
                    line = get_stats_line(stats, power)
                    energy_file.write(line)
                    energy_file.flush()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', 
                        help='path to logs',
                        )
    parser.add_argument('--energy-csv', 
                        help='csv file name',
                        default='energy.csv')
    parser.add_argument('--process-csv', 
                        help='csv file name',
                        default='processes.csv')
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_dir + "/monitoring.log", 
        level=logging.DEBUG,
        format='%(levelname)s - %(asctime)s - %(filename)s - %(lineno)d : %(message)s',
        )
    
    try:
        main(args)
    except Exception as err:
        logging.error(err)