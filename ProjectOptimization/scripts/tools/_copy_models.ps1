$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.." )).Path
$srcRoot = Join-Path $projectRoot "src\ultralytics-main"
$dstRoot = Join-Path $projectRoot "models"

if (Test-Path "$srcRoot\yolo11n-pose.pt") {
  Copy-Item "$srcRoot\yolo11n-pose.pt" "$dstRoot\yolo\pretrained\yolo11n-pose.pt" -Force
}
if (Test-Path "$srcRoot\yolo11n.pt") {
  Copy-Item "$srcRoot\yolo11n.pt" "$dstRoot\yolo\pretrained\yolo11n.pt" -Force
}
if (Test-Path "$srcRoot\runs\pose\train\weights\best.pt") {
  Copy-Item "$srcRoot\runs\pose\train\weights\best.pt" "$dstRoot\yolo\custom\yolo_pose_best.pt" -Force
}
if (Test-Path "$srcRoot\runs\pose\train\weights\last.pt") {
  Copy-Item "$srcRoot\runs\pose\train\weights\last.pt" "$dstRoot\yolo\custom\yolo_pose_last.pt" -Force
}
if (Test-Path "$srcRoot\models\st_gcn.kinetics.pt") {
  Copy-Item "$srcRoot\models\st_gcn.kinetics.pt" "$dstRoot\stgcn\pretrained\st_gcn.kinetics.pt" -Force
}

$legacyDir = "$srcRoot\work_dir-gcn\recognition\kinetics_skeleton\ST_GCN"
if (Test-Path $legacyDir) {
  Get-ChildItem $legacyDir -Filter "epoch*_model.pt" | ForEach-Object {
    Copy-Item $_.FullName "$dstRoot\stgcn\legacy\" -Force
  }
}

Write-Output "copy_models_done"
