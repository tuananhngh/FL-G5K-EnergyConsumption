#from jtop import jtop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', 
                    help='path to logs')
args = parser.parse_args()
print(args.log_dir)

def main():
    with jtop() as jetson:
        with open(args.log_dir+"tegrastats.csv","w+") as f:
            f.write(f"timestamp, RAM%, GPU%, GPU inst power, GPU avg power, CPU%, CPU inst power, CPU avg power, tot inst power, tot avg power\n")
            
            while jetson.ok():
                stats = jetson.stats
                power = jetson.power
                cpu = sum([stats['CPU'+str(i)] for i in range(1,9)])/8
                f.write(f"{stats['time']}, {stats['RAM']}, {stats['GPU']}, {power['rail']['GPU']['power']/100}, {power['rail']['GPU']['avg']/100}, {cpu}, {power['rail']['CPU']['power']/100}, {power['rail']['CPU']['avg']/100},{power['tot']['power']/100}, {power['tot']['avg']/100}\n")

if __name__ == "__main__":
    main()