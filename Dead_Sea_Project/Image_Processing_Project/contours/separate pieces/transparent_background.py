from PIL import Image


def transparent_background(img_name, mask_name):
    # load images
    img_org = Image.open('./' + img_name)
    img_mask = Image.open('./' + mask_name)

    # convert images
    # img_org  = img_org.convert('RGB') # or 'RGBA'
    img_mask = img_mask.convert('L')  # grayscale

    # the same size
    # img_org  = img_org.resize((400,400))
    # img_mask = img_mask.resize((400,400))

    # add alpha channel
    img_org.putalpha(img_mask)

    # save as png which keeps alpha channel
    img_org.save(img_name + '.png')


# transparent_background('M40975-1-E.jpg', 'mask_M40975-1-E.jpg')
