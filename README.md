# Andy Warhol Image Converter
The purpose of this web application is to convert a user-uploaded photo (.png, .jpg or .jpeg) into an Andy-Warhol inspired image. The user can chose the number of colours and the colours themselves and then download the image in .png format

## Web Application
Click [here](https://andy-warhol-image-converter-xctcvej4appkufhfyamypga.streamlit.app/) to go to the Andy Warhol Image Converter web appication.
## Source Code
Source code is available fo downlad and is composed of a single python script.
The process of converting the image is as follows:
1. Using 1000 pixels from the image, a k-means model is created to classify all pixels in the imageby colour similarity. RGB space is used.
2. The cluster of each pixel is predicted.
3. Each cluster is assigned a colour
4. Pixels are coloured according the the clouster they are classidied as

The following libraries are needed to run the code:
1. pandas
2. numpy
3. sklearn (train_test_split and kMeans modules)
4. random
5. streamlit
6. webcolors
7. PIL (image module)
8. io (BytesIO module)

## Known issues
When uplaoding the image, it is downsized considerably for comuptaitons to complete faster, this may be an issue for some users. Future updates will allow for the possiblity of downloading a larger version of the converted image.
On some ocassions, the second-to-last version of the image is downloaded, that is, if the user changes a colour of the image, that change will be visible on the web application but not on the downlaodd image. The image has to be downladed again.
