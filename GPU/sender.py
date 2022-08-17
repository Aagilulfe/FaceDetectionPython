#!/usr/bin/env python

from __future__ import division
from tabnanny import verbose
import cv2
import numpy as np
import socket
import struct
import math
import time

#Flag for print activation
verbose = False

class FrameSegment(object):
    """ 
    Object to break down image frame segment
    if the size of image exceed maximum datagram size 
    """
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64 # extract 64 bytes in case UDP frame overflown
    
    def __init__(self, sock, port, addr): #127.0.0.1
        self.s = sock
        self.port = port
        self.addr = addr
        self.ping = 0
        self.feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.feedback_socket.bind(("192.168.1.102", 12346))
    
    def receive_feedback(self):
        self.feedback_socket.settimeout(0.01)
        try:
            seg, _ = self.feedback_socket.recvfrom(self.MAX_DGRAM)
        except socket.timeout:
            print("!!!!timeout = no feedback")
            return
        return seg
    
    def send_ping_result(self): # 255 flag to indicate ping packet
        # print(self.ping)
        self.s.sendto(struct.pack("B", 255) + struct.pack("q", self.ping), (self.addr, self.port))
    
    def udp_frame(self, img):
        """ 
        Compress image and Break down
        into data segments 
        """
        compress_img = cv2.imencode('.jpg', img)[1]
        dat = compress_img.tobytes()
        size = len(dat)
        count = math.ceil(size/(self.MAX_IMAGE_DGRAM))
        array_pos_start = 0
        while count:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            if count == 1:
                timestamp = int(time.time()*1000)
                self.s.sendto(struct.pack("B", count) + struct.pack("q", timestamp) + 
                    dat[array_pos_start:array_pos_end], 
                    (self.addr, self.port)
                    )
                
                if verbose: print("waiting for feedback")
                seg = self.receive_feedback()
                if seg != None:
                    # print(seg)
                    new_timestamp = int(time.time() * 1000)
                    #print(struct.unpack("q", seg)[0], new_timestamp)
                    self.ping = (new_timestamp - struct.unpack("q", seg)[0]) // 2
                
            else:
                self.s.sendto(struct.pack("B", count) + #struct.pack("q", int(time.time()*1000)) + 
                    dat[array_pos_start:array_pos_end], 
                    (self.addr, self.port)
                    )
                
            array_pos_start = array_pos_end
            count -= 1


def main():
    """ Top level main function """
    # Set up UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port = 12345
    receiver_addr = "192.168.1.102"

    fs = FrameSegment(s, port, addr=receiver_addr)
    
    cap = cv2.VideoCapture(0)   #Webcam
    #cap = cv2.VideoCapture(1)   #ZED cam
    while (cap.isOpened()):
        _, frame = cap.read()
        if verbose: print("send frame")
        fs.udp_frame(frame)
        
        ping = fs.ping
        if verbose: print("send ping result\n")
        fs.send_ping_result()
        cv2.putText(frame, str(ping)+"ms", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        cv2.imshow("sender", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    s.close()

if __name__ == "__main__":
    main()
