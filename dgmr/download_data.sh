if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if METEO_FRANCE_DATA_PATH is defined
if [ -z "$METEO_FRANCE_DATA_PATH" ]; then
    # If not, set the default path to data
    METEO_FRANCE_DATA_PATH="data"
fi

# Create the directory if necessary
if [ ! -d "$METEO_FRANCE_DATA_PATH" ]; then
    mkdir -p "$METEO_FRANCE_DATA_PATH"
    echo "Directory $METEO_FRANCE_DATA_PATH created successfully."
fi

get_rounded_date() {
    current_minutes=$(date +%M)
    rounded_minutes=$(( (current_minutes) / 5 * 5 ))

    if [ $rounded_minutes -eq 60 ]; then
        rounded_minutes=0
        date +"%Y-%m-%d_%H-00" -d "+1 hour"
    else
        date +"%Y-%m-%d_%H-$rounded_minutes"
    fi
}

METEO_FRANCE_API_URL='https://public-api.meteofrance.fr/public/DPRadar/v1/mosaiques/METROPOLE/observations/LAME_D_EAU/produit?maille=500'

max_retries=20
retry_delay=3
retry_count=0
success=false

while [ $retry_count -lt $max_retries ]; do
    curl -X 'GET' $METEO_FRANCE_API_URL -H 'accept: application/octet-stream+gzip' -H "apikey: $METEO_FRANCE_API_KEY" --output "$METEO_FRANCE_DATA_PATH/$(get_rounded_date).h5"
    
    if [ $? -eq 0 ]; then
        success=true
        break
    else
        echo "Attempt $(($retry_count + 1)) failed. Retry in $retry_delay seconds..."
        retry_count=$(($retry_count + 1))
        sleep $retry_delay
    fi
done

if [ "$success" = true ]; then
    echo "Download successful."
else
    echo "Download fails after $max_retries attempts."
fi