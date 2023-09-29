import csv
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
from scipy import io
import random

def D2samplefunction():
    # LIFEFB:
    D2imgname = []
    D2mos_all = []
    D2sample = []
    root="/home/data/hrz/data/qgy/IQA-Dataset/liveFB"
    csv_file = os.path.join(root, "labels_image.csv")
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # D2imgname.append(row["name"])
            mos = np.array(float(row["mos"])).astype(np.float32)
            D2mos_all.append(mos)
            D2sample.append(os.path.join(root, "database", row["name"]))

    return D2sample

class KONIQDATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):
        super(KONIQDATASET, self).__init__()

        self.data_path = root
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "koniq10k_scores_and_distributions.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["image_name"])
                mos = np.array(float(row["MOS_zscore"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for _, item in enumerate(index):
            for _ in range(patch_num):
                sample.append(
                    (os.path.join(root, "1024x768", imgname[item]), mos_all[item])
                )

        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class LIVECDATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):

        imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
        imgpath = imgpath["AllImages_release"]
        imgpath = imgpath[7:1169]
        mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
        labels = mos["AllMOS_release"].astype(np.float32)
        labels = labels[0][7:1169]

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append(
                    (os.path.join(root, "Images", imgpath[item][0][0]), labels[item])
                )

        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class LIVEDataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):

        refpath = os.path.join(root, "refimgs")
        refname = getFileName(refpath, ".bmp")

        jp2kroot = os.path.join(root, "jp2k")
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, "jpeg")
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, "wn")
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, "gblur")
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, "fastfading")
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = io.loadmat(os.path.join(root, "dmos_realigned.mat"))
        labels = dmos["dmos_new"].astype(np.float32)

        orgs = dmos["orgs"]
        refnames_all = io.loadmat(os.path.join(root, "refnames_all.mat"))
        refnames_all = refnames_all["refnames_all"]

        refname.sort()
        sample = []

        for i in range(0, len(index)):
            train_sel = refname[index[i]] == refnames_all
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[0][item]))
                    # LQ_path = imgpath[item]
                    # HQ_path = os.path.join(root, 'refimgs', refnames_all[0][item][0])
                    # label = labels[0][item]
        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = "%s%s%s" % ("img", str(index), ".bmp")
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


class TID2013Dataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):
        refpath = os.path.join(root, "reference_images")
        refname = getTIDFileName(refpath, ".bmp.BMP")
        txtpath = os.path.join(root, "mos_with_names.txt")
        fh = open(txtpath, "r")
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split("\n")
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(
                        (
                            os.path.join(root, "distorted_images", imgnames[item]),
                            labels[item],
                        )
                    )
        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


class CSIQDataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):

        refpath = os.path.join(root, "src_imgs")
        refname = getFileName(refpath, ".png")
        txtpath = os.path.join(root, "csiq_label.txt")
        fh = open(txtpath, "r")
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split("\n")
            words = line[0].split()#1600.AWGN.1 0.062
            imgnames.append((words[0])+ "."+"png")#1600.AWGN.1
            target.append(words[1])#0.062
            ref_temp = words[0].split(".")#1600 AWGN 1
            refnames_all.append(ref_temp[0] + "." + "png")#1600.png

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(
                        (
                            os.path.join(root, "dst_imgs_all", imgnames[item]),
                            labels[item],
                        )
                    )
        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class KADIDDataset(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):
        refpath = os.path.join(root, "reference_images")
        # refname = getTIDFileName(refpath, ".png.PNG")
        refname = getFileName(refpath, ".png")

        imgnames = []
        target = []
        refnames_all = []

        csv_file = os.path.join(root, "dmos.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row["dist_img"])
                refnames_all.append(row["ref_img"])

                mos = np.array(float(row["dmos"])).astype(np.float32)
                target.append(mos)

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []
        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for _ in range(patch_num):
                    sample.append(
                        (
                            os.path.join(root, "images", imgnames[item]),
                            labels[item],
                        )
                    )
        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class SPAQDATASET(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None, transform_t=None):
        super(SPAQDATASET, self).__init__()

        self.data_path = root
        anno_folder = os.path.join(self.data_path, "Annotations")
        xlsx_file = os.path.join(anno_folder, "MOS and Image attribute scores.xlsx")
        read = pd.read_excel(xlsx_file)
        imgname = read["Image name"].values.tolist()
        mos_all = read["MOS"].values.tolist()
        for i in range(len(mos_all)):
            mos_all[i] = np.array(mos_all[i]).astype(np.float32)
        sample = []
        for _, item in enumerate(index):
            for _ in range(patch_num):
                sample.append(
                    (
                        os.path.join(
                            self.data_path,
                            "TestImage",
                            imgname[item],
                        ),
                        mos_all[item],
                    )
                )

        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class FBLIVEFolder(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None ,transform_t=None):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "labels_image.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["name"])
                mos = np.array(float(row["mos"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append(
                    (os.path.join(root, "database", imgname[item]), mos_all[item])
                )

        # D2sample = D2samplefunction()

        self.samples = sample
        # self.D2samples = D2sample
        self.transform = transform
        # self.transform_t = transform_t

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # D2path = self.D2samples[random.randint(0, len(self.D2samples) - 1)]
        # D2sample = self._load_image(D2path)
        path, target = self.samples[index]
        sample = self._load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
            # D2sample = self.transform_t(D2sample)

        return sample, sample, target

    def __len__(self):
        length = len(self.samples)
        return length
