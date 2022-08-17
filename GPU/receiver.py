#!/usr/bin/env python

from __future__ import division
import cv2
import numpy as np
import socket
import struct
import time

MAX_DGRAM = 2**16

def dump_buffer(s):
    """ Emptying buffer frame """
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        print(seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            print("finish emptying buffer")
            break

def resend_timestamp(s, timestamp, sender_addr):
    sender_port = 12346
    #print(timestamp)
    #print(struct.pack("q", timestamp))
    s.sendto(struct.pack("q", timestamp), (sender_addr, sender_port))
    
def receive_feedback():
    feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    feedback_socket.bind(("192.168.1.102", 12346))
    feedback_socket.settimeout(0.04)
    try:
        seg, _ = feedback_socket.recvfrom(MAX_DGRAM)
    except socket.timeout:
        pass
    return seg

def main(local_address):
    """ Getting image udp frame &
    concate before decode and output image """
    
    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((local_address, 12345))    #127.0.0.1
    dat = b''
    dump_buffer(s)
    ping = 0    # initialisation of ping

    while True:
        seg, sender_addr = s.recvfrom(MAX_DGRAM)
        if 1 < struct.unpack("B", seg[0:1])[0] < 255:
            #print(struct.unpack("B", seg[0:1]))
            #print(seg[1:50])
            #print(struct.unpack("q", seg[1:9]))
            dat += seg[1:]
        elif struct.unpack("B", seg[0:1])[0] == 255:
            # print(seg)
            ping = struct.unpack("q", seg[1:])[0]
            print(ping)
        else:
            #print(struct.unpack("B", seg[0:1]))
            #print(struct.unpack("q", seg[1:9]))
            timestamp = struct.unpack("q", seg[1:9])[0]
            
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
    main('192.168.1.102')