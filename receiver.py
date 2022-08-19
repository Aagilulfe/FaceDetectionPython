#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import argparse
import os


MAX_DGRAM = 2**16

def dump_buffer(s):
    """ Emptying buffer frame """
    print("emptying buffer...")
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        print(seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            print("-->finish emptying buffer\n")
            break

def resend_timestamp(s, timestamp, sender_addr):
    sender_port = 12346
    #print(timestamp)
    #print(struct.pack("q", timestamp))
    s.sendto(struct.pack("q", timestamp), (sender_addr, sender_port))

def main(listenning_addr, verbose):
    """ Getting image udp frame &
    concate before decode and output image """
    
    local_address = listenning_addr
    local_port = 12345
    
    os.system('cls')
    print("Receiver started\n")
    print("Listening on {} : {}\n===================================\n".format(local_address, local_port))

    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((local_address, local_port))    #127.0.0.1
    dat = b''
    dump_buffer(s)
    ping = 0    # initialisation of ping
    
    print("\n===>RUNNING\n")
    while True:
        if verbose: print("receiving packet")
        seg, sender_addr = s.recvfrom(MAX_DGRAM)
        if 1 < struct.unpack("B", seg[0:1])[0] < 255:
            if verbose: print("-> it's part of frame")
            #print(struct.unpack("B", seg[0:1]))
            #print(seg[1:50])
            #print(struct.unpack("q", seg[1:9]))
            dat += seg[1:]
        elif struct.unpack("B", seg[0:1])[0] == 255:
            if verbose: print("-> it's ping measure\n")
            # print(seg)
            ping = struct.unpack("q", seg[1:])[0]
            # print(ping)
        else:
            if verbose: print("-> it's end of frame")
            #print(struct.unpack("B", seg[0:1]))
            #print(struct.unpack("q", seg[1:9]))
            timestamp = struct.unpack("q", seg[1:9])[0]
            
            if verbose: print("--> resend the timestamp")
            resend_timestamp(s, timestamp, sender_addr[0])

            #ping = int(time.time()*1000) - timestamp
            dat += seg[9:]
            img = cv2.imdecode(np.frombuffer(dat, dtype=np.uint8), 1)
            cv2.putText(img, str(ping)+"ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('receiver', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            dat = b''

    # cap.release()
    cv2.destroyAllWindows()
    s.close()


if __name__ == "__main__":
    # init argument parser
    parser = argparse.ArgumentParser(description="=====UDP stream receiver=====\n", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--ip", 
        help="IP address to listen on", 
        default="192.168.1.102", 
        type=str
    )
    parser.add_argument('--verbose', help="enables print logs", action='store_true')
    
    verbose = parser.parse_args().verbose
    
    args = parser.parse_args()
    listenning_addr = args.ip

    main(listenning_addr, verbose)