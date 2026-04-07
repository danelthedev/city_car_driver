#include "ReShade.fxh"

// City Car Driving minimap cleaner + reprojector
// 1) Captures the top-right minimap from the backbuffer
// 2) Reduces semi-transparent background bleed with color classification
// 3) Separates road from environment
// 4) Recolors dark street-name text on roads to road color
// 5) Draws the cleaned minimap at another screen location

uniform float SourceX <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_label = "Source X";
    ui_tooltip = "Left edge of original minimap in normalized screen coordinates.";
> = 0.936;

uniform float SourceY <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_label = "Source Y";
    ui_tooltip = "Top edge of original minimap in normalized screen coordinates.";
> = 0.028;

uniform float SourceW <
    ui_type = "drag";
    ui_min = 0.02; ui_max = 0.4;
    ui_step = 0.001;
    ui_label = "Source Width";
    ui_tooltip = "Width of original minimap in normalized screen coordinates.";
> = 0.060;

uniform float SourceH <
    ui_type = "drag";
    ui_min = 0.02; ui_max = 0.4;
    ui_step = 0.001;
    ui_label = "Source Height";
    ui_tooltip = "Height of original minimap in normalized screen coordinates.";
> = 0.140;

uniform float DestX <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_label = "Destination X";
    ui_tooltip = "Left edge of cleaned minimap destination rectangle.";
> = 0.020;

uniform float DestY <
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.001;
    ui_label = "Destination Y";
    ui_tooltip = "Top edge of cleaned minimap destination rectangle.";
> = 0.760;

uniform float DestW <
    ui_type = "drag";
    ui_min = 0.05; ui_max = 0.7;
    ui_step = 0.001;
    ui_label = "Destination Width";
    ui_tooltip = "Width of cleaned minimap destination rectangle.";
> = 0.230;

uniform float DestH <
    ui_type = "drag";
    ui_min = 0.05; ui_max = 0.7;
    ui_step = 0.001;
    ui_label = "Destination Height";
    ui_tooltip = "Height of cleaned minimap destination rectangle.";
> = 0.230;

uniform float CleanupStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Cleanup Strength";
    ui_tooltip = "How strongly colors are pushed toward road/environment classes.";
> = 0.85;

uniform float RoadSatMax <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Road Saturation Max";
    ui_tooltip = "Lower values make road detection stricter (roads are low saturation).";
> = 0.38;

uniform float RoadLumMin <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Road Luminance Min";
    ui_tooltip = "Road luminance lower bound.";
> = 0.20;

uniform float RoadLumMax <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Road Luminance Max";
    ui_tooltip = "Road luminance upper bound.";
> = 0.85;

uniform float TextDarkThreshold <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Text Dark Threshold";
    ui_tooltip = "How dark a pixel must be to be treated as street-name text.";
> = 0.34;

uniform float TextEdgeBoost <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.05;
    ui_label = "Text Edge Boost";
    ui_tooltip = "Higher values detect thin text strokes more aggressively.";
> = 1.35;

uniform float TextSuppressStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Text Suppress Strength";
    ui_tooltip = "Higher values remove dark street-name text more aggressively.";
> = 1.55;

uniform float ResidualTextKill <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 2.0;
    ui_step = 0.05;
    ui_label = "Residual Text Kill";
    ui_tooltip = "Fallback cleanup for leftover dark text fragments on roads.";
> = 1.35;

uniform float RoadInpaintStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.05;
    ui_label = "Road Inpaint Strength";
    ui_tooltip = "Force-fills remaining dark road text using surrounding road pixels.";
> = 2.20;

uniform float RoadHoleFillStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 3.0;
    ui_step = 0.05;
    ui_label = "Road Hole Fill";
    ui_tooltip = "Fills any non-road speckles inside road neighborhoods.";
> = 2.40;

uniform float ForceRoadFill <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Force Road Fill";
    ui_tooltip = "Brutal mode: force road color for pixels inside strong road neighborhoods.";
> = 0.0;

uniform float ForceRoadThreshold <
    ui_type = "slider";
    ui_min = 0.30; ui_max = 0.95;
    ui_step = 0.01;
    ui_label = "Force Road Threshold";
    ui_tooltip = "Lower values affect more pixels; start around 0.62.";
> = 0.62;

uniform float MajorityDespeckle <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Majority Despeckle";
    ui_tooltip = "Removes tiny isolated specs by forcing the local majority class.";
> = 1.0;

uniform float BlueToRoadStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Blue To Road";
    ui_tooltip = "Converts blue minimap markings to drivable road color.";
> = 1.0;

uniform float BlueToRoadThreshold <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.6;
    ui_step = 0.01;
    ui_label = "Blue Threshold";
    ui_tooltip = "Lower values convert more blue shades to road color.";
> = 0.10;

uniform float GreenToRoadStrength <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Green Player To Road";
    ui_tooltip = "Converts green player arrow pixels to drivable road color.";
> = 1.0;

uniform float GreenToRoadThreshold <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.6;
    ui_step = 0.01;
    ui_label = "Green Threshold";
    ui_tooltip = "Lower values catch more green shades from the player arrow.";
> = 0.08;

uniform float HardClassify <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Hard Classification";
    ui_tooltip = "Pushes minimap pixels toward solid classes to remove transparency blending.";
> = 0.95;

uniform float4 RoadColor <
    ui_type = "color";
    ui_label = "Road Color";
> = float4(0.64, 0.66, 0.67, 1.0);

uniform float4 EnvironmentColor <
    ui_type = "color";
    ui_label = "Environment Color";
> = float4(0.18, 0.44, 0.19, 1.0);

uniform float4 WaterOrVoidColor <
    ui_type = "color";
    ui_label = "Water/Void Color";
> = float4(0.10, 0.20, 0.30, 1.0);

uniform float BorderThickness <
    ui_type = "slider";
    ui_min = 0.0; ui_max = 0.02;
    ui_step = 0.0005;
    ui_label = "Border Thickness";
> = 0.002;

uniform float4 BorderColor <
    ui_type = "color";
    ui_label = "Border Color";
> = float4(0.03, 0.03, 0.03, 1.0);

texture2D BackBufferTex : COLOR;
sampler2D BackBufferSampler
{
    Texture = BackBufferTex;
    MinFilter = LINEAR;
    MagFilter = LINEAR;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

float2 GetBufferTexelSize()
{
#if defined(BUFFER_RCP_WIDTH) && defined(BUFFER_RCP_HEIGHT)
    return float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);
#elif defined(BUFFER_WIDTH) && defined(BUFFER_HEIGHT)
    return float2(1.0 / max(BUFFER_WIDTH, 1.0), 1.0 / max(BUFFER_HEIGHT, 1.0));
#else
    // Conservative fallback for uncommon preset environments.
    return float2(1.0 / 1920.0, 1.0 / 1080.0);
#endif
}

float Luma(float3 c)
{
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float3 RGBToHSV(float3 c)
{
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = (c.g < c.b) ? float4(c.bg, K.wz) : float4(c.gb, K.xy);
    float4 q = (c.r < p.x) ? float4(p.xyw, c.r) : float4(c.r, p.yzx);

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float SampleEdgeStrength(float2 uv, float2 texel)
{
    float3 c = tex2D(BackBufferSampler, uv).rgb;
    float3 cx = tex2D(BackBufferSampler, uv + float2(texel.x, 0.0)).rgb;
    float3 cy = tex2D(BackBufferSampler, uv + float2(0.0, texel.y)).rgb;
    float3 dx = abs(cx - c);
    float3 dy = abs(cy - c);
    return Luma(dx + dy);
}

float RoadMaskFromColor(float3 c)
{
    float lum = Luma(c);
    float3 hsv = RGBToHSV(c);
    float roadBySat = 1.0 - smoothstep(RoadSatMax * 0.65, RoadSatMax, hsv.y);
    float roadByLum = smoothstep(RoadLumMin, RoadLumMin + 0.08, lum) * (1.0 - smoothstep(RoadLumMax - 0.08, RoadLumMax, lum));
    return saturate(roadBySat * roadByLum);
}

float3 ClassifyOpaqueColor(float3 c)
{
    float road = RoadMaskFromColor(c);
    float water = saturate((c.b - max(c.r, c.g)) * 3.5 + 0.2) * (1.0 - road);
    float env = saturate((c.g - max(c.r, c.b)) * 3.5 + 0.4) * (1.0 - road);

    float wRoad = max(road, 1e-5);
    float wEnv = max(env * (1.0 - water), 1e-5);
    float wWater = max(water, 1e-5);
    float wSum = wRoad + wEnv + wWater;

    float3 soft = (RoadColor.rgb * wRoad + EnvironmentColor.rgb * wEnv + WaterOrVoidColor.rgb * wWater) / wSum;

    float3 hard = EnvironmentColor.rgb;
    if (wRoad >= wEnv && wRoad >= wWater)
    {
        hard = RoadColor.rgb;
    }
    else if (wWater >= wRoad && wWater >= wEnv)
    {
        hard = WaterOrVoidColor.rgb;
    }

    return lerp(soft, hard, HardClassify);
}

float3 ClassWeights(float3 c)
{
    float road = RoadMaskFromColor(c);
    float water = saturate((c.b - max(c.r, c.g)) * 3.5 + 0.2) * (1.0 - road);
    float env = saturate((c.g - max(c.r, c.b)) * 3.5 + 0.4) * (1.0 - road);
    return float3(road, env * (1.0 - water), water);
}

float3 DominantClassColor(float3 w)
{
    float3 dominant = EnvironmentColor.rgb;
    if (w.x >= w.y && w.x >= w.z)
    {
        dominant = RoadColor.rgb;
    }
    else if (w.z >= w.x && w.z >= w.y)
    {
        dominant = WaterOrVoidColor.rgb;
    }
    return dominant;
}

float ColorBoxMask(float3 c, float3 key, float3 tol)
{
    float3 d = abs(c - key);
    return (1.0 - step(tol.r, d.r)) *
           (1.0 - step(tol.g, d.g)) *
           (1.0 - step(tol.b, d.b));
}

float3 CleanMinimapPixel(float2 srcUV, float2 texel)
{
    float3 c = tex2D(BackBufferSampler, srcUV).rgb;
    float lum = Luma(c);

    float3 cRight = tex2D(BackBufferSampler, srcUV + float2(texel.x, 0.0)).rgb;
    float3 cLeft = tex2D(BackBufferSampler, srcUV - float2(texel.x, 0.0)).rgb;
    float3 cUp = tex2D(BackBufferSampler, srcUV - float2(0.0, texel.y)).rgb;
    float3 cDown = tex2D(BackBufferSampler, srcUV + float2(0.0, texel.y)).rgb;
    float3 cRU = tex2D(BackBufferSampler, srcUV + float2(texel.x, -texel.y)).rgb;
    float3 cLU = tex2D(BackBufferSampler, srcUV + float2(-texel.x, -texel.y)).rgb;
    float3 cRD = tex2D(BackBufferSampler, srcUV + float2(texel.x, texel.y)).rgb;
    float3 cLD = tex2D(BackBufferSampler, srcUV + float2(-texel.x, texel.y)).rgb;
    float2 texel2 = texel * 2.0;
    float3 cRight2 = tex2D(BackBufferSampler, srcUV + float2(texel2.x, 0.0)).rgb;
    float3 cLeft2 = tex2D(BackBufferSampler, srcUV - float2(texel2.x, 0.0)).rgb;
    float3 cUp2 = tex2D(BackBufferSampler, srcUV - float2(0.0, texel2.y)).rgb;
    float3 cDown2 = tex2D(BackBufferSampler, srcUV + float2(0.0, texel2.y)).rgb;
    float3 cRU2 = tex2D(BackBufferSampler, srcUV + float2(texel2.x, -texel2.y)).rgb;
    float3 cLU2 = tex2D(BackBufferSampler, srcUV + float2(-texel2.x, -texel2.y)).rgb;
    float3 cRD2 = tex2D(BackBufferSampler, srcUV + float2(texel2.x, texel2.y)).rgb;
    float3 cLD2 = tex2D(BackBufferSampler, srcUV + float2(-texel2.x, texel2.y)).rgb;

    float roadMask = RoadMaskFromColor(c);
    float roadAround = 0.25 * (RoadMaskFromColor(cRight) + RoadMaskFromColor(cLeft) + RoadMaskFromColor(cUp) + RoadMaskFromColor(cDown));
    float roadAround8 = 0.125 * (
        RoadMaskFromColor(cRight) + RoadMaskFromColor(cLeft) + RoadMaskFromColor(cUp) + RoadMaskFromColor(cDown) +
        RoadMaskFromColor(cRU) + RoadMaskFromColor(cLU) + RoadMaskFromColor(cRD) + RoadMaskFromColor(cLD)
    );

    float3 classColor = ClassifyOpaqueColor(c);
    float3 classified = lerp(c, classColor, CleanupStrength);

    float3 w9 =
        ClassWeights(c) +
        ClassWeights(cRight) + ClassWeights(cLeft) + ClassWeights(cUp) + ClassWeights(cDown) +
        ClassWeights(cRU) + ClassWeights(cLU) + ClassWeights(cRD) + ClassWeights(cLD);
    float3 majorityColor = DominantClassColor(w9);

    float edge = SampleEdgeStrength(srcUV, texel) * TextEdgeBoost;
    float darkText = 1.0 - smoothstep(TextDarkThreshold - 0.08, TextDarkThreshold + 0.03, lum);
    float aroundLum = 0.25 * (Luma(cRight) + Luma(cLeft) + Luma(cUp) + Luma(cDown));
    float aroundLum8 = 0.125 * (Luma(cRight) + Luma(cLeft) + Luma(cUp) + Luma(cDown) + Luma(cRU) + Luma(cLU) + Luma(cRD) + Luma(cLD));
    float darkVsNeighbors = smoothstep(0.04, 0.18, aroundLum - lum);

    // Text is typically dark, thin, and mostly inside roads.
    float textMask = saturate((0.65 * darkText + 0.75 * darkVsNeighbors) * smoothstep(0.02, 0.14, edge) * max(roadMask, roadAround) * TextSuppressStrength);
    classified = lerp(classified, RoadColor.rgb, textMask);

    // Fallback killer for residual label fragments that survive edge-based detection.
    float roadCore = smoothstep(0.42, 0.72, max(roadMask, roadAround));
    float darkResidual = 1.0 - smoothstep(TextDarkThreshold + 0.02, TextDarkThreshold + 0.14, lum);
    float residualMask = saturate(roadCore * darkResidual * ResidualTextKill);
    classified = lerp(classified, RoadColor.rgb, residualMask);

    // Aggressive inpaint: if a dark pixel is surrounded by roads, force it to road color.
    float surroundedByRoad = smoothstep(0.58, 0.90, roadAround8);
    float darkComparedToArea = smoothstep(0.03, 0.20, aroundLum8 - lum);
    float inpaintMask = saturate(surroundedByRoad * darkComparedToArea * RoadInpaintStrength);
    classified = lerp(classified, RoadColor.rgb, inpaintMask);

    // Hole fill: catches residual artifacts that are not dark enough for text-specific tests.
    float3 avg8 = 0.125 * (cRight + cLeft + cUp + cDown + cRU + cLU + cRD + cLD);
    float centerVsRoad = length(classified - RoadColor.rgb);
    float avgVsRoad = length(avg8 - RoadColor.rgb);
    float looksLikeRoadNeighborhood = smoothstep(0.55, 0.92, roadAround8) * (1.0 - smoothstep(0.20, 0.46, avgVsRoad));
    float centerIsOutlier = smoothstep(0.03, 0.22, centerVsRoad - avgVsRoad);
    float holeFillMask = saturate(looksLikeRoadNeighborhood * centerIsOutlier * RoadHoleFillStrength);
    classified = lerp(classified, RoadColor.rgb, holeFillMask);

    // Final hard snap removes leftover semi-transparent blending from source minimap.
    classified = lerp(classified, classColor, 0.35 + 0.65 * CleanupStrength);

    // Re-apply inpaint after hard snap so tiny glyph traces cannot come back.
    classified = lerp(classified, RoadColor.rgb, inpaintMask);
    classified = lerp(classified, RoadColor.rgb, holeFillMask);

    // Guaranteed-effect mode for stubborn labels: overwrite road neighborhoods.
    float forceRoadMask = smoothstep(ForceRoadThreshold, min(ForceRoadThreshold + 0.20, 1.0), roadAround8) * ForceRoadFill;
    classified = lerp(classified, RoadColor.rgb, forceRoadMask);

    // Last-stage despeckle for single-pixel leftovers.
    float centerOutlier = smoothstep(0.025, 0.12, length(classified - majorityColor));
    float speckleMask = MajorityDespeckle * centerOutlier;
    classified = lerp(classified, majorityColor, speckleMask);

    // Preserve markers/icons exactly from source, but only via strict color keys.
    const float3 markerBlueKey = float3(0.11, 0.26, 0.44);
    const float3 markerBlueTol = float3(0.18, 0.18, 0.18);
    const float3 markerGreenKey = float3(0.1451, 0.9333, 0.1059); // #25EE1B
    const float3 markerGreenTol = float3(0.22, 0.22, 0.22);

    float blueSeedCenter = ColorBoxMask(c, markerBlueKey, markerBlueTol);
    float blueRight = ColorBoxMask(cRight, markerBlueKey, markerBlueTol);
    float blueLeft = ColorBoxMask(cLeft, markerBlueKey, markerBlueTol);
    float blueUp = ColorBoxMask(cUp, markerBlueKey, markerBlueTol);
    float blueDown = ColorBoxMask(cDown, markerBlueKey, markerBlueTol);
    float blueRU = ColorBoxMask(cRU, markerBlueKey, markerBlueTol);
    float blueLU = ColorBoxMask(cLU, markerBlueKey, markerBlueTol);
    float blueRD = ColorBoxMask(cRD, markerBlueKey, markerBlueTol);
    float blueLD = ColorBoxMask(cLD, markerBlueKey, markerBlueTol);
    float blueSeedNear = max(max(max(blueRight, blueLeft), max(blueUp, blueDown)), max(max(blueRU, blueLU), max(blueRD, blueLD)));

    float blueRight2 = ColorBoxMask(cRight2, markerBlueKey, markerBlueTol);
    float blueLeft2 = ColorBoxMask(cLeft2, markerBlueKey, markerBlueTol);
    float blueUp2 = ColorBoxMask(cUp2, markerBlueKey, markerBlueTol);
    float blueDown2 = ColorBoxMask(cDown2, markerBlueKey, markerBlueTol);
    float blueRU2 = ColorBoxMask(cRU2, markerBlueKey, markerBlueTol);
    float blueLU2 = ColorBoxMask(cLU2, markerBlueKey, markerBlueTol);
    float blueRD2 = ColorBoxMask(cRD2, markerBlueKey, markerBlueTol);
    float blueLD2 = ColorBoxMask(cLD2, markerBlueKey, markerBlueTol);
    float blueSeedFar = max(max(max(blueRight2, blueLeft2), max(blueUp2, blueDown2)), max(max(blueRU2, blueLU2), max(blueRD2, blueLD2)));

    float greenSeedCenter = ColorBoxMask(c, markerGreenKey, markerGreenTol);
    float greenRight = ColorBoxMask(cRight, markerGreenKey, markerGreenTol);
    float greenLeft = ColorBoxMask(cLeft, markerGreenKey, markerGreenTol);
    float greenUp = ColorBoxMask(cUp, markerGreenKey, markerGreenTol);
    float greenDown = ColorBoxMask(cDown, markerGreenKey, markerGreenTol);
    float greenRU = ColorBoxMask(cRU, markerGreenKey, markerGreenTol);
    float greenLU = ColorBoxMask(cLU, markerGreenKey, markerGreenTol);
    float greenRD = ColorBoxMask(cRD, markerGreenKey, markerGreenTol);
    float greenLD = ColorBoxMask(cLD, markerGreenKey, markerGreenTol);
    float greenSeedNear = max(max(max(greenRight, greenLeft), max(greenUp, greenDown)), max(max(greenRU, greenLU), max(greenRD, greenLD)));

    // For the player marker, only restore bright neighborhood pixels to avoid dark text bleed.
    float greenBrightNeighborhood = greenSeedNear * smoothstep(0.52, 0.92, lum);

    float blueMask = saturate(blueSeedCenter + blueSeedNear + 0.65 * blueSeedFar);
    float greenMask = saturate(greenSeedCenter + greenBrightNeighborhood);
    float markerMask = saturate(max(blueMask, greenMask));
    classified = lerp(classified, c, markerMask);

    return classified;
}

bool InRect(float2 uv, float2 p, float2 s)
{
    return uv.x >= p.x && uv.x <= (p.x + s.x) && uv.y >= p.y && uv.y <= (p.y + s.y);
}

float2 RectToUV(float2 uv, float2 rectPos, float2 rectSize)
{
    return (uv - rectPos) / max(rectSize, float2(1e-6, 1e-6));
}

float BorderMask(float2 uv, float2 pos, float2 size, float thickness)
{
    float2 local = RectToUV(uv, pos, size);
    float2 d = min(local, 1.0 - local);
    float inner = step(thickness / max(size.x, 1e-6), d.x) * step(thickness / max(size.y, 1e-6), d.y);
    return 1.0 - inner;
}

float4 PS_MinimapCleanup(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target
{
    float4 scene = tex2D(BackBufferSampler, uv);

    float2 srcPos = float2(SourceX, SourceY);
    float2 srcSize = float2(SourceW, SourceH);

    float2 dstPos = float2(DestX, DestY);
    float2 dstSize = float2(DestW, DestH);

    float4 outColor = scene;

    if (InRect(uv, dstPos, dstSize))
    {
        float2 local = RectToUV(uv, dstPos, dstSize);
        float2 srcUV = srcPos + local * srcSize;

        float2 texel = GetBufferTexelSize();
        float3 cleaned = CleanMinimapPixel(srcUV, texel);

        // Force a fully opaque overlay region in RGB space.
        outColor.rgb = cleaned;

        float border = BorderMask(uv, dstPos, dstSize, BorderThickness);
        outColor.rgb = lerp(outColor.rgb, BorderColor.rgb, border * BorderColor.a);
    }

    return outColor;
}

technique CCD_Minimap_Cleanup
{
    pass
    {
        VertexShader = PostProcessVS;
        PixelShader = PS_MinimapCleanup;
    }
}
