from PIL import Image
import os


def file_name(file_dir):
    os.makedirs(file_dir+'-128',exist_ok=True)
    for root, dirs, files in os.walk(file_dir):
        count = 1
        # 当前文件夹所有文件
        for i in files:
            # 判断是否以.jpg结尾
            if i.endswith('.png'):
                # 如果是就改变图片像素为640,480
                im = Image.open(os.path.join(file_dir,i))
                out = im.resize((128, 128))
                out.save(file_dir+'-128/' + str('wp') + str(count) + '.jpg', 'JPEG')
                count += 1
                print(i)
        break


file_name('sketches_all_resized')  # 当前文件夹
