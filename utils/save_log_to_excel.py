import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def write_excel(sheet, data_type, line, epoch, itr, loss, weight):
    sum_loss = 0
    if data_type == 'train':
        sheet.write(line, 0, epoch + 1)
        sheet.write(line, 1, itr + 1)
        for i in range(3):
            sheet.write(line, i + 2, round(loss[i], 6))
            sum_loss += loss[i] * weight[i]
        sheet.write(line, 5, round(sum_loss, 6))
        # sheet.write(line, 4, round(sum_loss, 6))
    elif data_type == 'val':
        loss, val, train = loss
        sheet.write(line, 0, epoch + 1)
        for i in range(3):
            sheet.write(line, i + 1, round(loss[i], 6))
        sheet.write(line, 4, round(val, 6))
        sheet.write(line, 5, round(train, 6))
    elif data_type == 'test':
        sheet.write(line, 0, epoch)
        for i in range(3):
            sheet.write(line, i + 1, round(loss[i], 6))
            sum_loss += loss[i] * weight[i]
        sheet.write(line, 4, round(sum_loss, 6))
    return line + 1


def init_excel(kind):
    workbook = xlwt.Workbook()
    if kind == 'train':
        sheet1 = workbook.add_sheet('train', cell_overwrite_ok=True)
        sheet2 = workbook.add_sheet('val', cell_overwrite_ok=True)
        # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
        row0 = ["epoch", "itr", "l2", "ssim", "vgg", "loss"]
        # row0 = ["epoch", "itr", "l2", "ssim", "loss"]
        row1 = ["epoch", "l2", "ssim", "vgg", "val_loss", "train_loss"]
        # row1 = ["epoch", "l2", "ssim", "val_loss", "train_loss"]
        for i in range(0, len(row0)):
            print('写入train_excel')
            sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        for i in range(0, len(row1)):
            print('写入val_excel')
            sheet2.write(0, i, row1[i], set_style('Times New Roman', 220, True))
        return workbook, sheet1, sheet2
    elif kind == 'test':
        sheet1 = workbook.add_sheet('test', cell_overwrite_ok=True)
        # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
        row0 = ["num", "l2", "ssim", "vgg", "sum_loss"]
        for i in range(0, len(row0)):
            print('写入test_excel')
            sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        return workbook, sheet1
