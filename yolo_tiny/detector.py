import torch


class Detector:
    def __init__(self, net_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = torch.load(net_path, map_location=self.device.type)
        self.net.eval()

    def detect(self, x_in, thresh, anchors):
        out13, out26 = self.net(x_in.to(self.device))

        # keep boxes over thresh
        idxs26, vectors26 = self.boxFilter(out26, thresh)
        idxs13, vectors13 = self.boxFilter(out13, thresh)

        # calculate the real position
        boxes26 = self.boxReturn(idxs26, vectors26, 16, anchors[26]).to(self.device)
        boxes13 = self.boxReturn(idxs13, vectors13, 32, anchors[13]).to(self.device)

        return torch.cat([boxes26, boxes13], dim=0)

    def boxFilter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size()[0], output.size()[1],
                                output.size()[2], 3, -1)  # N, H, W, 3, 5 (iou, cx, cy, w, h, cls)

        mask = output[..., 0] > thresh
        idxs = mask.nonzero()
        vectors = output[mask]
        return idxs, vectors

    def boxReturn(self, idxs, vectors, scaleRate, anchors):
        if vectors.size()[0] == 0:
            return torch.tensor([])

        # vectors [iou, cx_offset, cy_offset, w_offset, h_offset]
        anchors = torch.tensor(anchors, dtype=torch.float32).to(self.device)
        batch, idx_cy, idx_cx, boxType = idxs[:, 0], idxs[:, 1], idxs[:, 2], idxs[:, 3]  # [N, H, W, 3(Box Type), 10]
        cx, cy = (idx_cx.float() + vectors[:, 1]) * scaleRate, (idx_cy.float() + vectors[:, 2]) * scaleRate
        w, h = anchors[boxType, 0] * torch.exp(vectors[:, 3]), anchors[boxType, 1] * torch.exp(vectors[:, 4])
        confi = vectors[..., 0]
        cls0 = vectors[..., 5]
        cls1 = vectors[..., 6]
        cls2 = vectors[..., 7]
        cls3 = vectors[..., 8]
        cls4 = vectors[..., 9]

        return torch.stack([batch.float(), confi, cx, cy, w, h, cls0, cls1, cls2, cls3, cls4], dim=1)
