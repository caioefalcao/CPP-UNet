# CPP-UNet: Combined Pyramid Pooling Modules in the U-Net Network for Kidney, Tumor and Cyst Segmentation

<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://kits-challenge.org/public/site_media/figures/axial_new.png" alt="Markdownify" width="400"></a>
</h1>
<h2> 
  CPP-UNet, an innovative convolutional neural network-based architecture designed for the segmentation of renal structures, including the kidneys themselves and renal masses (cysts and tumors), in a computed tomography (CT) scan. Particularly, we investigate the fusion of the Pyramid Pooling Module (PPM) and Atrous Spatial Pyramid Pooling (ASPP) for improving the UNet network by integrating contextual information across multiple scales.
  
  </h2>

## CPP-NET results applied to KiTS21 and KiTS23 datasets
  
<table class="tg"><thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax">KiTS21 - Tumor Dice</th>
    <th class="tg-0lax">KiTS23 - Tumor Dice</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0lax">UNet (Standard)</td>
    <td class="tg-baqh">83,59%</td>
    <td class="tg-baqh">88,08%</td>
  </tr>
  <tr>
    <td class="tg-0lax">PPM-Deeplabv3+</td>
    <td class="tg-baqh">85,11%</td>
    <td class="tg-baqh">83,59%</td>
  </tr>
  <tr>
    <td class="tg-1wig">CPP-UNet (Our)</td>
    <td class="tg-amwm">85,69%</td>
    <td class="tg-amwm">88,17%</td>
  </tr>
</tbody>
</table>