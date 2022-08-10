from matplotlib.style import use
from Detector import Detector
import argparse


if __name__ == "__main__":

    # init argument parser
    parser = argparse.ArgumentParser(description="Face detection with OpenCV and sending to client with Gstreamer.\nCredits: Agilulfe", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--local-display", 
        help="Display on local machine or not : 0 (default) or 1", 
        default=0, 
        type=int
    )
    
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="IP to stream to",
        type=str
    )
    
    parser.add_argument(
        "--port",
        default=5000,
        help="Port to stream to",
        type=int
    )
    
    parser.add_argument(
        "--flip",
        default=0,
        help="Flip method : 0 (default), 1, 2 or 3",
        type=int
    )
    
    parser.add_argument(
        "--batch",
        default=1,
        help="Batch size (the lower the slower)",
        type=int
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or gpu",
        type=ascii
    )

    # parsing script arguments
    args = parser.parse_args()
    local_display = bool(args.local_display)
    CLIENT_IP = args.ip
    CLIENT_PORT = args.port
    flip = args.flip
    batch = args.batch
    if batch > 20:
        batch = 20
    device = args.device
    if batch <= 0:
        raise Exception("Batch size must be greater than 0")

    # output passed arguments
    print("CONFIGURATION:")
    print("- Ip: ", CLIENT_IP)
    print("- Port: ", CLIENT_PORT)
    print("- Display locally? ", local_display)
    print("- Flip: ", flip)
    print("- Batch size: ", batch)
    print("- Device: ", device)
    print("RUNNING... (Press 'q' to quit)")
    print("\n==============================")  

    if device == "cpu":
        use_cuda = False
    else:
        use_cuda = True

    # create the detector
    detector = Detector(CLIENT_IP, CLIENT_PORT, local_display, flip, batch, use_cuda)

    print("\nWhich method of face detection do you want to use?")
    print("[0] Image processing\n[1] Video processing locally\n[2] Video processing with UDP sending")
    print("[q] QUIT")
    method = input("\nYour choice: ")
    
    if method == "0":
        imgName = input("Name of the image: ")
        detector.processImage(imgName)
    if method == "1":
        detector.processVideo()
    if method == "2":
        detector.processVideo()