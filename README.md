
<img src="ascii-kore_projects.png" alt="(ASCII) Kore Projects" style="width: 100%" />
<div align="center">
  <h1 style="margin-bottom: 0;">ASCII Wallpapers</h1>
    <div style="margin-top: 0;">
        <br>
        <a href="https://gitlab.com/Korvexx">
            <img src="https://img.shields.io/badge/made%20by-Korvexx-pink" alt="GitLab" />
        </a>
        <a href="https://ko-fi.com/korvexx">
            <img src="https://img.shields.io/badge/consider%20donating%20💖-white" alt="Donating" />
        </a>
    </div>
    </br>
    <p>
        ASCII Wallpapers is a wallpaper addon for KDE Plasma transforming any static image into ASCII-Art.<br>
        Choose your font and character set, adjust detail-level and grayscale and switch between image and slideshow mode!
    </p>
    <br>
</div>

<h3>Preview of the addon</h3>
<div align="center">
<table style="border: none;">
<tr>
<!-- Low res-->
<td align="center"><img src="screenshots/wallpaper_lowres.png" width="400"/>

<strong>Low Detail</strong></td>
<!-- Medium res-->
<td align="center"><img src="screenshots/wallpaper_mediumres.png" width="400"/>

<strong>Medium Detail</strong></td>
</tr>
<tr>

<!-- High res-->
<td align="center"><img src="screenshots/wallpaper_highres.png" width="400"/>

<strong>High Detail</strong></td>

<!-- Grayscaled-->
<td align="center"><img src="screenshots/wallpaper_grayscaled.png" width="400"/>

<strong>Grayscale Mode</strong></td>
</tr>
</table>


<img src="screenshots/settings.png" width="600"/>


<em>The Wallpaper Settings Configuration Page</em>
</div>

<h3>How does it work?</h3>
<div>
    <p>
    Using a GLSL shader, the wallpaper is converted to ASCII in real-time.<br>
    The steps to achieve the ASCII-fication are quite simple:
    </p>
    <ul>
        <li><strong>Atlas Generation:</strong> When the wallpaper is loaded, a hidden `canvas` element draws the sequence of ASCII characters from darkest to lightes in the selected detail size onto a texture strip which is called an <strong>Atlas</strong>.</li>
        <li><strong>Grid Sampling:</strong> The shader divides the screen into a grid which is based on the chosen <strong>Detail Level</strong>. It samples the color of the original image at the center of each grid cell to determine the color for that character</li>
        <li><strong>Intensity Mapping:</strong> The shader calculates the brightness of the sampled color acting as an index to pick the most appropriate character from the Atlas.</li>
        <li><strong>Final Render:</strong> Each grid cell is masked by the chosen character. If <strong>Grayscale</strong> is enabled, the original color is desaturated before being multiplied by the character mask, resulting in the final output.</li>
    </ul>
</div>

<h3>How to use it</h3>
<div>
    <ol>
        <li>Navigate: <strong>System settings > Wallpaper</strong></li>
        <li>Set <strong>Wallpaper type</strong> to "ASCII Wallpaper"</li>
        <li>Select a Wallpaper</li>
        <li>Adjust <strong>Detail level</strong> and <strong>Grayscale</strong> if wanted</li>
        <li>Click apply</li>
    </ol>
</div>


<h3>Donations</h3>
<p>
I develop projects like <strong>ASCII Wallpapers</strong> in my free time, so if you enjoy this addon and want to support me and my projects, consider donating so that I can dedicate more time to mainting my projects and creating new ones!
</p>

<div align="center">
  <a href="https://ko-fi.com/H2H3YHHOC"><img src="https://img.shields.io/badge/Donate%20on-Ko--Fi-magenta" alt="Ko-Fi" /></a>
  <a href="https://liberapay.com/Korvexx/donate"><img src="https://img.shields.io/badge/Donate%20on-Liberapay-yellow" alt="Liberapay" /></a>
  <a href="https://www.paypal.com/donate/?hosted_button_id=R32URW6RPXB24"><img src="https://img.shields.io/badge/Donate%20on-Paypal-blue" alt="Paypal" /></a>
</div>
</div>
