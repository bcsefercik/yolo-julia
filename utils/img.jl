import Images
import Images: N0f8, RGB, Normed
import ImageDraw
import Plots

include("constants.jl")

shown_img_type = Array{RGB{Normed{UInt8,8}},2}

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
        img = ImageDraw.draw(img, p, LABEL_COLOR)
    end

    return img
end

function draw_results(x, y_pred, y_gold=nothing, font_size=10)
    A = permutedims(x, (3,1,2))
    img = Images.colorview(RGB, A)

    img = convert(shown_img_type, img);

    y_pred_int = map(o -> Integer.(round.(o)), y_pred)

    for i in 1:length(y_pred)
        r = y_pred_int[i]
        x1, y1, x2, y2 = r[2], r[3], r[4], r[5]

        pol = get_polygon(x1, y1, x2, y2)

        ImageDraw.draw!(img, pol, COLORS[r[1]])
    end

    if y_gold != nothing
        img = draw_labels(img, y_gold)
    end

    p = Plots.plot(img, size=(416, 416), axis=nothing)

    for i in 1:length(y_pred)
        r = y_pred_int[i]
        x1, y1 = r[2], r[3]
        Plots.annotate!(p,
            x1,
            y1 - font_size - 2,
            Plots.text("█████", :white, :left, "verdana", font_size + 2)
        )
        Plots.annotate!(
            p,
            x1 + 2,
            y1 - font_size,
            Plots.text(
                CLASS_NAMES_R[r[1]],
                COLORS[r[1]],
                :left,
                "verdana",
                font_size
            )
        )
    end

    return p
end
