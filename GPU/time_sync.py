import time
import os


def synchronize():
    print("Synchronizing...")
    try:
        import ntplib
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        os.system('sudo date ' + time.strftime('%m%d%H%M%Y.%S',time.localtime(response.tx_time)))
    except:
        print('Could not sync with time server.')

    print('Done.')

if __name__ == "__main__":
    synchronize()