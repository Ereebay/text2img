import os
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *
import pickle
import imageio
import skimage

dataset = '200birds'

if dataset == '102flowers':
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
    # 设定当前目录，img目录，caption目录，以及vocab目录
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '102flowers/102flowers')
    caption_dir = os.path.join(cwd, '102flowers/text_c10')
    VOC_FIR = cwd + '/vocab.txt'

    # load captions
    caption_sub_dir = load_folder_list(caption_dir)
    captions_dict = {}  # 存放拆分的单词
    processed_capts = []  # 存档处理后的语句
    for sub_dir in caption_sub_dir:  # get caption file list
        with suppress_stdout():
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])  # 获得caption的中对应图像的index
                t = open(file_dir, 'r')
                lines = []
                for line in t:
                    line = preprocess_caption(line)  # 对每一行caption进行预处理
                    lines.append(line)  # 处理完的caption装载到lines中
                    processed_capts.append(
                        tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))  # 将line中的词拆开装到processed_capts
                assert len(lines) == 10, "Every flower image have 10 captions"  # 判断每个图是否有10个captions
                captions_dict[key] = lines  # 在对应索引的caption_dict中装在文本内容
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))  # 打印caption总数 图像数量*10

    ## build vocab
    if not os.path.isfile('vocab.txt'):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)  # 建立vocab
    else:
        print("WARNING: vocab.txt already exists")
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")  # 将文字转换成ID

    ## store all captions ids in list
    captions_ids = []  # 放单词索引
    try:  # python3
        tmp = captions_dict.items()
    except:  # python3
        tmp = captions_dict.iteritems()
    # 记录单词在vocab中的索引 存入capitons_ids
    for key, value in tmp:
        for v in value:
            captions_ids.append(
                [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
            # print(v)              # prominent purple stigma,petals are white inc olor
            # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
            # exit()
    # captions_ids转成ndarry
    captions_ids = np.asarray(captions_ids)
    print(" * tokenized %d captions" % len(captions_ids))

    ## check
    img_capt = captions_dict[1][1]  # 将dict中的key为1的index为1的元素幅给img_capt
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]  # img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

    ## load images
    with suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))  # 读取图像文件的list
    print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
    s = time.time()

    # time.sleep(10)
    # def get_resize_image(name):   # fail
    #         img = scipy.misc.imread( os.path.join(img_dir, name) )
    #         img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
    #         img = img.astype(np.float32)
    #         return img
    # images = tl.prepro.threading_data(imgs_title_list, fn=get_resize_image)
    images = []
    images_256 = []
    for name in imgs_title_list:
        # print(name)
        img_raw = imageio.imread(os.path.join(img_dir, name))  # 读入原图
        img = tl.prepro.imresize(img_raw, size=[64, 64])  # (64, 64, 3) 对图像预处理为64*64
        img = img.astype(np.float32)  # 将图像类型转为float32
        images.append(img)  # 拼接到images中
    # images = np.array(images)
    # images_256 = np.array(images_256)
    print(" * loading and resizing took %ss" % (time.time() - s))

    n_images = len(captions_dict)  # 图像总数
    n_captions = len(captions_ids)  # captions总数
    n_captions_per_image = len(lines)  # 10

    print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

    captions_ids_train, captions_ids_test = captions_ids[: 8000 * n_captions_per_image], captions_ids[
                                                                                         8000 * n_captions_per_image:]  # 正序装入idstrain 倒序装入test
    images_train, images_test = images[:8000], images[8000:]
    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

elif dataset == '200birds':

    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '200birds/200birds_box')
    caption_dir = os.path.join(cwd, '200birds/text_c10')
    VOC_FIR = cwd + '/vocab.txt'
    TRAIN_CLASS_FILE = os.path.join(cwd, '200birds/trainvalclasses.txt')
    TEST_CLASS_FILE = os.path.join(cwd, '200birds/testclasses.txt')


    train_class_set = set()
    with open(TRAIN_CLASS_FILE, 'r') as f:
        for line in f:
            train_class_set.add(line.rstrip())

    test_class_set = set()
    with open(TEST_CLASS_FILE, 'r') as f:
        for line in f:
            test_class_set.add(line.rstrip())

    caption_sub_dir = sorted(load_folder_list( caption_dir ))
    processed_capts = []
    train_captions_list = []
    test_captions_list = []

    for sub_dir in caption_sub_dir: # get caption file list
        with suppress_stdout():
            files = sorted(tl.files.load_file_list(path=sub_dir, regx='_[0-9]+_[0-9]+\.txt'))
            basename = os.path.basename(sub_dir)

            if basename in train_class_set:
                target_list = train_captions_list
            elif basename in test_class_set:
                target_list = test_captions_list
            # else:
            #     raise KeyError

            for f in files:
                file_dir = os.path.join(sub_dir, f)
                t = open(file_dir,'r')

                line_num = 0
                for line in t:
                    preprocessed_line = preprocess_caption(line)
                    processed_capts.append(tl.nlp.process_sentence(preprocessed_line, start_word="<S>", end_word="</S>"))
                    target_list.append(preprocessed_line)
                    line_num += 1

                assert line_num == 10, "Every bird image should have 10 captions"


    ## build vocab
    if not os.path.isfile('vocab.txt'):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
    else:
        print("WARNING: vocab.txt already exists")
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")


    ## create captions id
    train_captions_ids = []
    for caption in train_captions_list:
        train_captions_ids.append(
            [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(caption)])
    train_captions_ids = np.asarray(train_captions_ids)

    test_captions_ids = []
    for caption in test_captions_list:
        test_captions_ids.append(
            [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(caption)])
    test_captions_ids = np.asarray(test_captions_ids)

    print(" * tokenized %d captions for training" % len(train_captions_ids))
    print(" * tokenized %d captions for testing" % len(test_captions_ids))
    print(" * tokenized %d captions total" % (len(train_captions_ids) + len(test_captions_ids)))

    ## check
    img_capt = train_captions_list[0]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

    del train_captions_list
    del test_captions_list

    ## load images
    s = time.time()

    train_image = []
    test_image = []
    train_image_256 = []
    test_image_256 = []
    image_sub_dir = sorted(load_folder_list(img_dir))
    for sub_dir in image_sub_dir:  # get image file list
        with suppress_stdout():  # get image files list
            files = sorted(tl.files.load_file_list(path=sub_dir, regx='_[0-9]+_[0-9]+\.jpg'))
            basename = os.path.basename(sub_dir)

            if basename in train_class_set:
                target_list = train_image
            elif basename in test_class_set:
                target_list = test_image
            # else:
            #     raise KeyError

            for f in files:
                img_raw = scipy.misc.imread(os.path.join(sub_dir, f))
                if len(img_raw.shape) < 3:
                    img_raw = skimage.color.gray2rgb(img_raw)
                img = tl.prepro.imresize(img_raw, size=[64, 64])  # (64, 64, 3)
                img = img.astype(np.float32)
                target_list.append(img)

    # train_image = np.asarray(train_image)
    # test_image = np.asarray(test_image)
    # train_image_256 = np.asarray(train_image_256)
    # test_image_256 = np.asarray(test_image_256)

    print(" * %d images for training" % len(train_image))
    print(" * %d images for testing" % len(test_image))
    print(" * %d images for total" % (len(train_image) + len(test_image)))

    print(" * loading and resizing took %ss" % (time.time()-s))

    n_captions_per_image = 10

    captions_ids_train, captions_ids_test = train_captions_ids, test_captions_ids
    images_train, images_test = train_image, test_image

    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)


save_all(vocab, '_vocab.pickle')
save_all(images_train, '_image_train.pickle')
save_all(images_test, '_image_test.pickle')
save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
