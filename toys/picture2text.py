"""
Python好玩的小项目之
将一个图片转换成字符画
"""

from PIL import Image

def transform(image_file):
    #生成字符画所需的字符集
    vocabulary = '''@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. '''
    text_image = ''
    #遍历每一个像素，'0'为横向大小，'1'为纵向
    for h in range(0,image_file.size[1]):
        for w in range(0,image_file.size[0]):
            g,r,b = image_file.getpixel((w,h)) # 返回每个像素对应的grb的值
            gray = int(r*0.299 + g*0.587 + b*0.114) # 转换成0~255的灰度
            text_image += vocabulary[int(((len(vocabulary)-1)*gray)/256)]
        text_image += '\r\n'
    return text_image



if __name__ == '__main__':
    image_file_dir = 'aaa.jpeg'
    text_file_dir = 'text_image.txt'
    with open(image_file_dir,'rb') as file:
        image_file = Image.open(file)
        #调整图片大小
        new_size = (int(image_file.size[0]*0.3), int(image_file.size[1]*0.2))
        image_file = image_file.resize(new_size)
        text_image = transform(image_file)

    with open(text_file_dir,'w') as text_file:
        text_file.write(text_image)
