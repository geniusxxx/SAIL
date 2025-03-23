# recap数据集内容和格式

## 原始wds数据集：

我现在有的原始cc3m_wds文件内部格式如下：一个uid对应一张图片，一个caption txt文件，一个json文件,json文件内容为：

```json
{
    "caption": "a river has burst it 's banks and has spread out onto arable farmland alongside",
    "url": "https://ak7.picdn.net/shutterstock/videos/8592247/thumb/1.jpg",
    "key": "000000000",
    "status": "success",
    "error_message": null,
    "width": 852,
    "height": 480,
    "exif": "{}",
    "original_width": 852,
    "original_height": 480
}；
```
## csv/parquet文件

对于`data_preparation/cc3m_3long_3short_1raw_captions_url.csv`，运行

```bash
❯ head -n 2 cc3m_3long_3short_1raw_captions_url.csv

Image Path,raw_caption,shortIB_captions,longIB_captions,shortSV_captions,longSV_captions,shortLLA_captions,longLLA_captions

https://s-media-cache-ak0.pinimg.com/736x/99/57/38/995738352e7f0fa677c5b082fc58540a.jpg,how to build an industry for dollars,a small cabin being built in the middle of a field,"In the image, there is a small black house with a green roof situated in a grassy area surrounded by trees. The house appears to be under construction or renovation, as there are various tools and materials visible around it, such as a hammer, nails, screws, and wood planks. The presence of these objects indicates that the house is being built or repaired, and the green roof adds a unique and eco-friendly feature to the structure.",A wooden building with a green roof is under construction.,"The image captures a tranquil scene in a wooded area. Dominating the frame is a small wooden cabin, its green roof contrasting with the surrounding foliage. The cabin's walls are painted a dark brown, and a window with white shutters punctuates one side. A door, also white, is situated on the opposite side. The cabin is elevated on a wooden platform, providing a vantage point over the verdant landscape. A hose, coiled and ready for use, lies on the ground in front of the cabin, hinting at the possibility of water sources nearby. The cabin is nestled amidst nature, with trees and bushes forming a lush backdrop. The precise location of the cabin is not discernible from the image, but its elevated position and the surrounding greenery suggest a peaceful retreat, possibly in a rural or semi-rural setting. There are no discernible texts or countable objects in the image. The relative positions of the objects are such that the cabin is the central focus, with the hose in the foreground and the trees and bushes in the background. The image does not provide any information that allows for a confident determination of object actions or precise object locations beyond what has been described.",A small wooden building is being constructed with a green roof.,"The image features a small wooden cabin with a green roof, surrounded by a grassy area. The cabin appears to be under construction, as there are several tools and materials scattered around the scene. A person is standing near the cabin, possibly working on the construction or observing the progress.   In addition to the cabin, there are two cars parked in the background, one on the left side and the other on the right side of the image. A bench can be seen in the middle of the scene, and a chair is located closer to the right side of the image."
```

分别为一个图像url和七个增强的captions。

## cc3m_recap_wds数据集

对于我要创建的cc3m_recap_wds数据集，根据分析，格式应该为：整体是wds格式的数据集，内部：一个uid对应一张图片和一个json，json文件为：

```json
{
    "raw_caption": [...],  // 原始caption
    "shortIB_captions": [...],  // 短的基于指令的caption
    "longIB_captions": [...],  // 长的基于指令的caption
    "shortSV_captions": [...],  // 短的监督生成caption
    "longSV_captions": [...],  // 长的监督生成caption
    "shortLLA_captions": [...],  // 短的LLaMa生成caption
    "longLLA_captions": [...],   // 长的LLaMa生成caption
    "url": "https://ak7.picdn.net/shutterstock/videos/8592247/thumb/1.jpg",
    "key": "000000000",
    "status": "success",
    "error_message": null,
    "width": 852,
    "height": 480,
    "exif": "{}",
    "original_width": 852,
    "original_height": 480
}；
```

## 如何利用已经下载好的cc3m_wds数据集创建cc3m_recap_wds数据集？

- 首先是图像的对应，我想到的是原始的cc3m_wds数据集每个图像对应的json文件中都有对应的url，而`data_preparation/cc3m_3long_3short_1raw_captions_url.csv`中的每行七个描述都有一个对应的url在image path，所以可以用这个来找到下载好的图片和csv中的七个描述的对应关系

- 需要删除cc3m_wds的所有txt文件，删除json文件中的"caption"，并向json文件中添加"raw_caption","shortIB_captions","longIB_captions","shortSV_captions","longSV_captions","shortLLA_captions","longLLA_captions"

- 分析代码可知，json中各caption字段格式为：

    ```json
        {
            "raw_caption": ["how to build an industry for dollars"],  // 单元素数组
            "shortIB_captions": ["a small cabin being built in the middle of a field"],
            "longIB_captions": ["In the image, there is a small black house with..."],
            "shortSV_captions": ["A wooden building with a green roof is under construction."],
            "longSV_captions": ["The image captures a tranquil scene in a wooded area..."],
            "shortLLA_captions": ["A small wooden building is being constructed with a green roof."],
            "longLLA_captions": ["The image features a small wooden cabin with a green roof..."]
        }
    ```

# yfcc15m-parquet数据集转换

## 原始yfcc15m数据集

- 格式：parquet

    - images：原始图片

    - texts：原始文本的token embedding，可利用tokenizer还原成原始文本

## yfcc15m-recap-wds数据集

要把原始的yfcc15m数据集转换成新的yfcc15m-recap-wds数据集

- 格式：wds

    - 一张图片

    - json文件：

        ```json
            {
                "raw_caption": ["how to build an industry for dollars"],  // 单元素数组
                "shortIB_captions": ["a small cabin being built in the middle of a field"],
                "longIB_captions": ["In the image, there is a small black house with..."],
                "shortSV_captions": ["A wooden building with a green roof is under construction."],
                "longSV_captions": ["The image captures a tranquil scene in a wooded area..."],
                "shortLLA_captions": ["A small wooden building is being constructed with a green roof."],
                "longLLA_captions": ["The image features a small wooden cabin with a green roof..."]
            }
        ```

## 转换步骤

目前的`yfcc15m_to_wds.py`代码实现的是将原始yfcc15m-parquet转换为wds格式，即得到原始图片+用tokenizer还原的原始文本

- 根据我的推测，原始文本应该就是新数据集的raw_caption，对于原数据集：原始文本与图片对应；对于新数据集的parquet文件`yfcc15m_3long_3short_1raw_captions_url.parquet`（里面是图片的url和对应的recap的七个captions），新文本与url对应，而新闻本中的raw_caption就是原始数据集的caption，因此可以建立映射关系

- 这样就可以直接把原始数据集的图片和新数据集的七个captions组成的json文件对应起来，构建新的yfcc15m-recap-wds数据集。