# Crowd Couting models using deep learning

## Topics

- State of the Art (Available models, available datasets)
- Implementation of some model
  - Test with images from testset
  - Test with images from TUM Campus

## Datasets
- ShanghaiTech: https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view
- UCF-QNRF
- UCF-CC-50
- Mall dataset: http://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html#:~:text=The%20mall%20dataset%20was%20collected,every%20pedestrian%20in%20all%20frames.


## TUM Testimages
- https://www.tum.de/fileadmin/_processed_/f/6/csm_20181015_Immatfeier_AH_485520-HDR-1600_abf4b6dcbb.jpg
- https://www.in.tum.de/fileadmin/w00bws/in/1.Fuer_Studieninteressierte/2.Fuer_Schuelerinnen_und_Schueler/abitagveranstaltung.jpg
- https://media.nature.com/lw800/magazine-assets/d42473-019-00093-9/d42473-019-00093-9_16565094.jpg

## Outlook
- Counting people in videos (tracking) - what are the challenges

## Resources

- Multiple models: https://paperswithcode.com/task/crowd-counting
- Multiple resources: https://paperswithcode.com/dataset/shanghaitech
- Example implementation in pytorch: https://www.analyticsvidhya.com/blog/2019/02/building-crowd-counting-model-python/
- Example using tensorflow https://github.com/darpan-jain/crowd-counting-using-tensorflow
- Second tensorflow CSRNet: https://github.com/Neerajj9/CSRNet-keras
- Counting penguins example: https://www.vision-systems.com/non-factory/article/14169857/deep-learning-algorithms-use-densitybased-crowd-counting-to-track-penguin-populations
- Benchmark for multi-object tracking (Leal-Taixe): https://arxiv.org/abs/2003.09003
- Work related to multi-object tracking from Leal Taixe: https://dvl.in.tum.de/team/lealtaixe/

- Comparison of algorithms: https://iopscience.iop.org/article/10.1088/1742-6596/1187/4/042012/pdf

- CSRNet real-time paper: https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050920X00093/1-s2.0-S187705092031053X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEOr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDwFQZiZgkaVFZRq2z2PU7dy0GAfG9F%2FUGQ1EK4YD1fGwIhAOFbNJA3ODsuJ75R3Rc5%2B%2F6zjuqO4MD2Z%2BleHYbHfeaEKoMECML%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBBoMMDU5MDAzNTQ2ODY1IgxO6c%2BQOxdeqRH7dBwq1wN%2Fhh7tYc9fDUo%2BbiGlDtbIDwSJ%2F2c1rqWn3FX7oX1Qgd5YQsViBN1yMlL9YZGWmRFc49v3WRjHj1RVWS2Fm5BNRQvAMYzzxJmHV9Rrf06vBRKWaSlpQLvVev%2B8YNAEjpPnitg%2Bw5%2BsPYPAXT2tWwcqyhStJLuTECBtdJHz8fn%2BZdMI8kjpAC8s2bR6R2u5KEM8QlxM3KTIuPsqLVhNFnci4olQpaZ3YzE3HhHSCyWoeVH6t2QcNPxOWHg5qmhN7BJ0ZUlEbPZS2edgWyymV0oDs5MRCFfFc0vowmkRECfCspwOWAU%2BFRox0WUfUF0qQBwiDiuFPzYiTPfF1bO60ZWkm5%2BLjuf%2BkMA5gFiB55ipYFyMQxJHOHlVSEyPBtiHpkHhZE42TGvoCi4g3yVV%2FMbz3XW5q%2BhvMXEwIZhkRuU0%2FzJRN4BG3%2FcmSJPZNCEefi7a1OoNfC%2FYK3WvdzAiPWG9AFKshbKeSWB49vYgOBIQx3lWh3BdmSnWc8ETStSuN%2FZ3SeHTxuCYAi6kiimctWwMLNZw321Gfhu2cSupJp%2BzLRg1SHDs8paPMk9aIgZrx1uSfwyMiKy3PCoI3nDbCqFp92Vd3y4JwfWd6IOolhTevbThBdpYobwwhe73hgY6pAEb8cqAPX6XnyQ2u4%2B1j5eRuHR84VLiRP%2BKTTqCFzriaqJHdgI4tL%2BmixQ0pjfX4srC9GA2nrpUVlaTWNenJBP210itCtTpC7Iy6dT3IGj64hjjq5bO9L5LkcO4t4AB9ey0KsLpGldVrSPHa%2Fc4rZEThDpHQx8T8ig9%2Bt%2FeKfWLt1mi1%2FdUmw4pLSSgYeWfxtWkPSQ2dOaLjsO8xLnUw30kDB%2FoiA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210701T174708Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2GIFZSFA%2F20210701%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=70b66725a1dee5e99473a033a35e1f2cd867d4d440a52d042ea1f3058a6ac430&hash=6f33838a4ffd9e2b8caf641e730b6e2dcc67a727d31f70f56eec12ac508b3977&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S187705092031053X&tid=spdf-4b36f69b-6420-4e5b-b0ce-dd76f38afc3c&sid=ef6cd3299756c043ac784889303254cb41abgxrqb&type=client

- Crowd-counting evaluation: https://www.researchgate.net/profile/David-Ryan-25/publication/264559856_An_Evaluation_of_Crowd_Counting_Methods_Features_and_Regression_Models/links/5e229b6592851cafc38c83b6/An-Evaluation-of-Crowd-Counting-Methods-Features-and-Regression-Models.pdf