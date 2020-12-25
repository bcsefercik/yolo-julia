import Images
import ImageDraw

function get_image_and_labels(name, labels, images_path="."; img_size=nothing)
    img = Images.load(joinpath(images_path, "$name.jpg"));

    if img_size != nothing
        img = Images.imresize(img, img_size)
    end

    return img, labels[name]
end

function get_x1y1x2y2(label, img_size=(416, 416))
    x1 = Integer(round((label[2] - label[4]/2) * img_size[2]))
    y1 = Integer(round((label[3] - label[5]/2) * img_size[1]))
    x2 = Integer(round((label[2] + label[4]/2) * img_size[2]))
    y2 = Integer(round((label[3] + label[5]/2) * img_size[1]))

    return (x1, y1, x2, y2)
end

function get_polygon(x1, y1, x2, y2)
    return ImageDraw.Polygon([(x1, y1), (x2,y1), (x2, y2), (x1, y2)])
end


function get_polygons(labels; img_size=(416, 416))
    return [get_polygon(get_x1y1x2y2(l, img_size)...) for l in labels]
end

function draw_labels(img, labels; img_size=(416, 416))
    if isa(img, String)
        img = Images.load(img);
    end

    if isa(labels, String)
        labels = JSON.parsefile(
            labels;
            dicttype=Dict,
            inttype=Integer,
            use_mmap=true
        )
    end

    if size(img) != img_size
        img = Images.imresize(img, img_size)
    end

    polygons = get_polygons(labels; img_size=img_size)

    for p in polygons
        img = ImageDraw.draw(img, p)
    end

    return img
end
