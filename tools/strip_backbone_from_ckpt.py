# tools/strip_backbone_from_ckpt.py
import torch, sys
src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv)>2 else src.replace('.pth', '_no_backbone.pth')
ckpt = torch.load(src, map_location='cpu')
state = ckpt.get('state_dict', ckpt)
new_state = {k:v for k,v in state.items() if not k.startswith('img_backbone.')}
if 'state_dict' in ckpt:
    ckpt['state_dict'] = new_state
else:
    ckpt = new_state
torch.save(ckpt, dst)
print('Saved:', dst, ' | kept params:', len(new_state))
