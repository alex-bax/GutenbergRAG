$resourceGroup = "rag"
$appName = "gbRAGFastAPI"

$settingsJson = Get-Content "envWebAppSettings.json" -Raw | ConvertFrom-Json

$settingsArgs = $settingsJson.PSObject.Properties | ForEach-Object {
    "$($_.Name)=$($_.Value)"
}

az webapp config appsettings set `
    --resource-group $resourceGroup `
    --name $appName `
    --settings $settingsArgs
