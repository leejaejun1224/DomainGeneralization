import cv2


def combine(syn_img, real_img, save_path, ratio):
    left = cv2.imread(syn_img)
    right = cv2.imread(real_img)
    
    h, w, c = left.shape
    h1, w1, c1 = right.shape
    
    crossbar = int(w*ratio)
    
    combined = left.copy()[:h-100,:]
    
    combined[:h-100, crossbar:] = right[:h-100, crossbar:]
    
    cv2.imwrite(save_path, combined)
    cv2.imshow("combined", combined)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__=="__main__":
    number = 658
    synthetic_img = "/home/jaejun/Downloads/gta/media/disk2/taesung/up2p/gta2cityscape_medres/trainA/00%d.png" % (number)
    real_img = "/home/jaejun/Downloads/images/00%d.png" % (number)
    save_path = "/home/jaejun/Downloads/combined/00%d.png" % (number)

    combine(syn_img=synthetic_img, real_img=real_img, save_path=save_path, ratio=0.52)
    