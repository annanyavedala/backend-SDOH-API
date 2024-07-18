import io
import os
import openpyxl
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jsonify.convert import jsonify
from sqlalchemy import create_engine, Column, String, Integer, func, MetaData, Table, DateTime, PrimaryKeyConstraint, or_, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Dict
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
import uuid
import geopandas as gpd
import json
import numpy as np

from starlette.responses import JSONResponse

app = FastAPI()
DATABASE_URL = "DATABASE URL"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()
metadata.create_all(engine)

origins = [
    "http://127.0.0.1:*",
    "http://localhost:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


class DataSource(Base):
    __tablename__ = "data_source"
    __table_args__ = {"schema": "data"}

    data_source_uuid = Column(String, primary_key=True)
    organization_id = Column(String)
    dataset_name = Column(String, primary_key=True)
    dataset_year = Column(String, primary_key=True)
    orgname_abbr = Column(String)
    datasetname_abbr = Column(String)
    organization_name = Column(String)
    domains = Column(String)
    variables_count = Column(Integer)
    boundary_type = Column(String)
    temporal_scale = Column(String)
    accessibility = Column(String)
    dataset_status = Column(String)
    phase = Column(String)
    variables_index_name = Column(String)


class IndexTable(Base):
    __tablename__ = 'dynamic_table_name'  # Initial placeholder name
    __table_args__ = {"schema": "data"}

    variable_id = Column(String, primary_key=True)
    variable_name = Column(String)
    variable_desc = Column(String)
    variable_value_type = Column(String)
    variable_value_unit = Column(String)
    data_source_uuid = Column(String)
    variable_concept_id = Column(String)

    @classmethod
    def set_tablename(cls, tablename):

        cls.__table__.name = tablename


def generate_variable_uuid(data_source_uuid, x, year):
    return (data_source_uuid + (2018 - year)) * 1000 + x + 1


def generate_dataset_uuid(db: Session, dataset_name: str, dataset_year: str):
    existing_dataset = db.query(DataSource).filter(
        DataSource.dataset_name == dataset_name,
        DataSource.dataset_year == dataset_year
    ).first()

    if existing_dataset:
        return existing_dataset.data_source_uuid

    last_dataset = db.query(DataSource).order_by(DataSource.data_source_uuid.desc()).first()
    last_uuid = last_dataset.data_source_uuid
    new_uuid = str(int(last_uuid) + 1)

    return new_uuid

    return new_uuid


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def fetch_data_from_database(db) -> List[Dict]:
    data = db.query(DataSource.dataset_name,
                    DataSource.dataset_year,
                    DataSource.organization_name,
                    DataSource.domains,
                    DataSource.variables_count,
                    DataSource.boundary_type,
                    DataSource.temporal_scale,
                    DataSource.accessibility,
                    DataSource.dataset_status,
                    DataSource.phase,
                    DataSource.variables_index_name).all()
    return [dict(row) for row in data]


def print_data():
    db = next(get_db())
    try:
        data = fetch_data_from_database(db)
        print(data)
    finally:
        db.close()





def sanitize_data(df):
    # Replace Inf and -Inf with NaN, then replace NaN with None
    return df.replace([np.inf, -np.inf, np.nan], None)


'''
Example json request body:
[{
    "year":"2016",
    "domain":"", 
    "dataset":"CDC/ATSDR Social Vulnerability Index (SVI)", 
    "variable":"m_totpop", 
    "state":"", 
    "county":""
}]
'''
@app.post("/get-census-tract-geojson")
async def get_census_tract_geojson(data: List[Dict], db: Session = Depends(get_db)):
    try:
        for item in data:
            dataset_year = item['year']
        # Initialize an empty GeoDataFrame
        table_name = f"census_tract_{dataset_year}"
        query = f"SELECT * FROM data.\"{table_name}\" limit 1"  # Only select necessary columns
        print('Read')

        gdf_temp = gpd.read_postgis(query, engine, geom_col='geom_local_value', crs='EPSG:4326')
    

        # Convert the merged GeoDataFrame to GeoJSON
        geojson = gdf_temp.to_json()
        geojson_dict = json.loads(geojson)
        
        features = geojson_dict['features']

        # Extract the properties of each feature into a list of dictionaries
        properties_list = [feature['properties'] for feature in features]

        # Create a DataFrame from the list of properties dictionaries
        features_df = pd.DataFrame(properties_list)

        # Rename the 'geocode' column to 'geoid'
        features_df.rename(columns={'geocode': 'geom_id'}, inplace=True)
        features_df['geom_id'] = features_df['geom_id'].apply(lambda x: str(x).zfill(11))
        print('Done')


        for item in data:
            variable = item['variable']
            dataset_name = item['dataset']
            dataset_year = item['year']

            matching_rows = db.query(DataSource).filter(
                DataSource.dataset_name == dataset_name,
                DataSource.dataset_year == dataset_year
            ).all()

            if not matching_rows:
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Accessing individual row attributes
            for row in matching_rows:
                index_name = str(row.variables_index_name)[:-5]
                print(index_name)
                variable_table_name = index_name + 'variables'
                variable_query = f"SELECT geocode, {variable} FROM data.\"{variable_table_name}\""
                variable_df = pd.read_sql_query(variable_query, db.bind)
                variable_df.rename(columns={'geocode':'geom_id'}, inplace=True)
                variable_df['geom_id'] = variable_df['geom_id'].apply(lambda x: str(x).zfill(11))
                


                variable_df = sanitize_data(variable_df)
               

            # Merge variable data into the features DataFrame
            features_df = features_df.merge(variable_df, on='geom_id', how='right')

        # Sanitize the merged DataFrame too
        features_df = sanitize_data(features_df)
        print(features_df.columns)

        # Update the GeoJSON properties with the new variable data
        for feature, new_data in zip(geojson_dict['features'], features_df.to_dict('records')):
            feature['properties'].update(new_data)

        # Serialize to JSON with safe floating-point handling
        # return json.dumps(geojson_dict)
        updated_geojson_path = f"updated_geojson_{dataset_year}.json"

        with open(updated_geojson_path, 'w') as f:
            json.dump(geojson_dict, f, indent=4)

        return {"message": f"GeoJSON updated and saved as {updated_geojson_path}"}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''
Example json request body:

[{
    "year":"2023", 
    "domain":"", 
    "dataset":"Area Health Resources Files (county)ENV", 
    "variable":"area_mi2_20", 
    "state":"", 
    "county":""
}]
'''
@app.post("/get-county-geojson")
async def get_county_geojson(data: List[Dict], db: Session = Depends(get_db)):
    try:
    
        # Load existing GeoJSON
        with open(f"geojson_county.json", 'r') as file:
            geojson = json.load(file)


        # Prepare a DataFrame from the GeoJSON features
        features_df = pd.DataFrame([feature['properties'] for feature in geojson['features']])
        features_df['geoid'] = features_df['GEOID']
        print('Done')



        for item in data:
            variable = item['variable']
            dataset_name = item['dataset']
            dataset_year = item['year']

            matching_rows = db.query(DataSource).filter(
                DataSource.dataset_name == dataset_name,
                DataSource.dataset_year == dataset_year
            ).all()

            if not matching_rows:
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Accessing individual row attributes
            for row in matching_rows:
                index_name = str(row.variables_index_name)[:-5]
                print(index_name)
                variable_table_name = index_name + 'variables'


            

            # Load variable data from PostgreSQL

            variable_query = f"SELECT geocode, {variable} FROM data.\"{variable_table_name}\""
            variable_df = pd.read_sql(variable_query, db.bind)
            variable_df.rename(columns={'geocode':'geoid'}, inplace=True)

            # Sanitize the data to remove NaN, Inf, -Inf
            variable_df = sanitize_data(variable_df)

            # Merge variable data into the features DataFrame
            features_df = features_df.merge(variable_df, on='geoid', how='left')

        # Sanitize the merged DataFrame too
        features_df = sanitize_data(features_df)

        # Update the GeoJSON properties with the new variable data
        for feature, new_data in zip(geojson['features'], features_df.to_dict('records')):
            feature['properties'].update(new_data)

        # Serialize to JSON with safe floating-point handling
        return json.dumps(geojson)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''
Example json request body:

[{
    "year":"2019", 
    "domain":"", 
    "dataset":"Crime in the United States, by State", 
    "variable":"MSA_POP", 
    "state":"", 
    "county":""
}]

'''
@app.post("/get-state-geojson")
async def get_state_geojson(data: List[Dict], db: Session = Depends(get_db)):
    try:
    
        # Load your existing GeoJSON
        with open(f"geojson_state.json", 'r') as file:
            geojson = json.load(file)


        # Prepare a DataFrame from the GeoJSON features
        features_df = pd.DataFrame([feature['properties'] for feature in geojson['features']])
        features_df['geoid'] = [feature['id'] for feature in geojson['features']]
        print(features_df.columns)
        print('Done')



        for item in data:
            variable = item['variable']
            dataset_name = item['dataset']
            dataset_year = item['year']

            matching_rows = db.query(DataSource).filter(
                DataSource.dataset_name == dataset_name,
                DataSource.dataset_year == dataset_year
            ).all()

            if not matching_rows:
                raise HTTPException(status_code=404, detail="Dataset not found")

            # Accessing individual row attributes
            for row in matching_rows:
                index_name = str(row.variables_index_name)[:-5]
                print(index_name)
                variable_table_name = index_name + 'variables'


            

            # Load variable data from PostgreSQL
            variable_query = f"SELECT Geocode, {variable} FROM data.\"{variable_table_name}\""
            variable_df = pd.read_sql(variable_query, db.bind)
            variable_df.rename(columns={'Geocode':'geoid'}, inplace=True)
            variable_df['geoid'] = variable_df['geoid'].apply(lambda x: str(x).zfill(2))

            # Sanitize the data to remove NaN, Inf, -Inf
            variable_df = sanitize_data(variable_df)

            # Merge variable data into the features DataFrame
            features_df = features_df.merge(variable_df, on='geoid', how='left')

        # Sanitize the merged DataFrame too
        features_df = sanitize_data(features_df)

        # Update the GeoJSON properties with the new variable data
        for feature, new_data in zip(geojson['features'], features_df.to_dict('records')):
            feature['properties'].update(new_data)

        # Serialize to JSON with safe floating-point handling
        return json.dumps(geojson)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






'''
No request body
'''
@app.get("/datasets")
async def get_datasets(db: Session = Depends(get_db)):
    print('Hello')
    data = fetch_data_from_database(db)

    domain_data = {'SDOH': [], 'Natural Environment': [], 'Geography': []}
    for row in data:
        domain_name = row["domains"]
        domain_data[domain_name].append(row)

    formatted_data = []
    for domain_name, rows in domain_data.items():
        unique_dataset_names = set(row["dataset_name"] for row in rows)
        domain_entry = {
            "domainName": domain_name,
            "data": []
        }
        for dataset_name in unique_dataset_names:
            rows_for_dataset = [row for row in rows if row["dataset_name"] == dataset_name]
            available_years = available_years = [row["dataset_year"] for row in rows_for_dataset]

            dataset_entry = {
                "Dataset Name": dataset_name,
                "Organization Name": rows_for_dataset[0]["organization_name"],
                "Boundary Type": rows_for_dataset[0]["boundary_type"],
                "Temporal Scale": rows_for_dataset[0]["temporal_scale"],
                "Accessibility": rows_for_dataset[0]["accessibility"],
                "Dataset Status": rows_for_dataset[0]["dataset_status"],
                "Phase": rows_for_dataset[0]["phase"],
                "Available Years": available_years,
                "Index table name": rows_for_dataset[0]['variables_index_name']
            }

            domain_entry["data"].append(dataset_entry)
        formatted_data.append(domain_entry)

    return {'data': formatted_data}




'''
No request json
'''
@app.get('/api/map')
def get_dataset_formatted(db: Session = Depends(get_db)):
    data = fetch_data_from_database(db)
    structured_data={"years":{}, "domains":{}}
    for row in data:
        year = row["dataset_year"]
        domain = row["domains"]

        # Handle Years
        if year not in structured_data["years"]:
            structured_data["years"][year] = {}
        if domain not in structured_data["years"][year]:
            structured_data["years"][year][domain] = []
        structured_data["years"][year][domain].append(row["dataset_name"])

        # Handle Domains (simplified for demonstration; adjust according to your exact requirements)
        if domain not in structured_data["domains"]:
            structured_data["domains"][domain] = {"years": [], "subdomains": [], "datasets": []}
        if year not in structured_data["domains"][domain]["years"]:
            structured_data["domains"][domain]["years"].append(year)
    return structured_data
        





'''
Example json request body:

http://127.0.0.1:8000/get-index-table/?dataset_name=AHRF Diversity Dashboard Data&dataset_year=2021

'''
@app.get("/get-index-table/")
def get_index(dataset_name: str, dataset_year: str, db: Session = Depends(get_db)):
    matching_rows = db.query(DataSource).filter(
        DataSource.dataset_name == dataset_name,
        DataSource.dataset_year == dataset_year
    ).all()

    if not matching_rows:
        raise HTTPException(status_code=404, detail="Dataset not found")

    merged_data = []

    for matching_row in matching_rows:
        custom_table_name = matching_row.variables_index_name
        table = Table(custom_table_name, metadata, autoload_with=engine, schema="data")

        with engine.connect() as connection:
            result = connection.execute(table.select())
            index_data = result.fetchall()

        formatted_data = [
            {
                "variable_id": row.variable_id,
                "variable_name": row.variable_name,
                "variable_desc": row.variable_desc,
                "variable_value_type": row.variable_value_type,
                "variable_value_unit": row.variable_value_unit,
                "data_source_uuid": row.data_source_uuid,
                "variable_concept_id": row.variable_concept_id
            }
            for row in index_data
        ]

        merged_data.extend(formatted_data)

    return {'data': merged_data}


'''
Example json request body:

{
  "sourceTable": {
      "org_name": "fwrwerw",
      "org_name_abbr": "werwer",
      "dataset_name": "aaa",
      "dataset_name_abbr": "werewr",
      "temporal_scale": "3-month",
      "accessibility": "werwer",
      "domain": "SDOH",
      "dataset_year":"2021"
  },
  "datasetTable": {
      "geoCode": "st_abbrev",
      "date": "",
      "time": "",
      "primaryKey": ""
  },
  "indexTable": {
      "variableName": "variableName",
      "description": "column4"
  }
}

'''
@app.post("/api/insert-data-source/")
def insert_data(item: Dict[str, Dict[str, str]], db: Session = Depends(get_db)):
    source_table = item.get("sourceTable", {})
    dataset_table = item.get("datasetTable", {})
    index_table = item.get("indexTable", {})

    # Extract data from sourceTable
    org_name = source_table.get("org_name", "")
    dataset_year = source_table.get("dataset_year", "")
    org_name_abbr = source_table.get("org_name_abbr", "")
    dataset_name = source_table.get("dataset_name", "")
    datasetname_abbr = source_table.get("dataset_name_abbr", "")
    temporal_scale = source_table.get("temporal_scale", "")
    accessibility = source_table.get("accessibility", "")
    domain = source_table.get("domain", "")

    # Generate unique dataset UUID
    data_source_uuid = generate_dataset_uuid(db, dataset_name, dataset_year)

    existing_entry = db.query(DataSource).filter(DataSource.data_source_uuid == data_source_uuid).first()
    if existing_entry:
        return {"status": 409, "message": "Dataset name already exisits."}

    # Check if organization name already exists
    existing_org = db.query(DataSource).filter(DataSource.organization_name == org_name).first()

    if existing_org:
        dataset_organization_id = existing_org.organization_id
    else:
        max_org_id = db.query(func.max(DataSource.organization_id)).scalar()
        if max_org_id is None:
            max_org_id = 0
        dataset_organization_id = int(max_org_id) + 1

    # Create and add the new dataset entry
    db_item = DataSource(
        organization_name=org_name,
        dataset_year=dataset_year,
        orgname_abbr=org_name_abbr,
        dataset_name=dataset_name,
        datasetname_abbr=datasetname_abbr,
        temporal_scale=temporal_scale,
        accessibility=accessibility,
        domains=domain,
        organization_id=dataset_organization_id,
        data_source_uuid=data_source_uuid
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    print('One done')

    # return db_item

    #######Inserting data into the variable index table#######
    variable_column_name = index_table.get("variableName", "")
    description_column_name = index_table.get("description", "")

    df = pd.read_csv('./index_table.csv')  # Update this line with the path to your file

    # Construct the table name
    custom_table_name = f"{dataset_name}_{dataset_year}_INDEX_TEST"

    metadata = MetaData()
    your_table = Table(
        custom_table_name,
        metadata,
        Column("variable_id", String),
        Column("variable_name", String),
        Column("description", String),
        Column("variable_value_type", String),
        Column("variable_value_unit", String),
        Column("data_source_uuid", String),
        Column("variable_concept_id", String),
        schema='data'
    )

    metadata.create_all(engine, checkfirst=True)

    with engine.connect() as conn:
        for index, row in df.iterrows():
            variable_uuid = generate_variable_uuid(int(data_source_uuid), index, int(dataset_year))
            variable_name = row[variable_column_name]
            description = row[description_column_name]

            conn.execute(
                your_table.insert().values(
                    variable_id=variable_uuid,
                    variable_name=variable_name,
                    description=description,
                    data_source_uuid=data_source_uuid
                )
            )

    print('Two done')
    ############# Insert variable table ################
    custom_table_name = f"{dataset_name}_{dataset_year}_VARIABLE_TEST"
    geocode = dataset_table.get("geoCode", "")
    date_column = dataset_table.get("date", "")
    time_column = dataset_table.get("time", "")
    pk = dataset_table.get("primaryKey", "")

    df = pd.read_csv('./dataset_table.csv')

    custom_table_name = f"{dataset_name}_{dataset_year}_VARIABLE_TEST"

    if len(date_column) <= 0:
        m = [0] * len(df)
        n = [0] * len(df)
        for index, row in df.iterrows():
            start_date = pd.to_datetime(dataset_year).replace(month=1, day=1, hour=0, minute=0, second=0)
            end_date = pd.to_datetime(dataset_year).replace(month=12, day=31, hour=23, minute=59, second=59)

            m[index] = start_date
            n[index] = end_date

        df["effective_start_date"] = m
        df["effective_end_date"] = n

    else:
        m = [0] * len(df)
        n = [0] * len(df)
        # Check the length of the date values and update 'effective_start_date' and 'effective_end_date' accordingly
        for index, row in df.iterrows():
            date_value = row[date_column]
            if len(date_value) > 4:  # Year
                start_date = pd.to_datetime(date_value).replace(month=1, day=1, hour=0, minute=0, second=0)
                end_date = pd.to_datetime(date_value).replace(month=12, day=31, hour=23, minute=59, second=59)
            else:
                start_date = pd.to_datetime(date_value).replace(hour=0, minute=0, second=0)
                end_date = pd.to_datetime(date_value).replace(hour=23, minute=59, second=59)

            m[index] = start_date
            n[index] = end_date

        df["effective_start_date"] = m
        df["effective_end_date"] = n

        # Rename the geoCode column to match the provided column name
    df['Geocode'] = df[geocode]

    # Create the SQLAlchemy Table object dynamically
    metadata = MetaData()
    columns = [
        Column(column_name, String)  # You can modify the data type based on the actual data
        for column_name in df.columns
    ]

    dataset_table = Table(
        custom_table_name,
        metadata,
        *columns,
        PrimaryKeyConstraint('effective_start_date', 'effective_end_date', 'Geocode'),
        schema='data'
    )

    # Create the table in the database
    metadata.create_all(engine)

    # Insert data into the table
    with engine.connect() as conn:
        df.to_sql(custom_table_name, conn, schema='data', if_exists='append', index=False)

        return {"status": 200, "message": "All datasets inserted successfully into the database."}


# ===== =====

async def read_data(file: UploadFile):
    try:
        # Reset file pointer to the start
        await file.seek(0)

        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file.file)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")

        # Close the file after reading
        await file.close()

        return df

    except ValueError as e:
        # Handle specific error (e.g., unsupported file format)
        raise e
    except Exception as e:
        # Handle other errors (e.g., file read errors)
        raise Exception(f"An error occurred while processing the file: {str(e)}")


async def read_excel_two(file: UploadFile):
    await file.seek(0)  # Reset file pointer to the beginning
    contents = await file.read()  # Read file content asynchronously
    file_content = io.BytesIO(contents)  # Create a BytesIO object
    workbook = openpyxl.load_workbook(file_content, data_only=True)  # Load workbook
    return workbook


@app.post("/api/tools/ingestion-tool/upload-files")
async def update_database_file(files: List[UploadFile] = File(...), filenames: List[str] = Form(...)):

    print(filenames)
    if not files:
        return JSONResponse(content={'error': 'No file provided'}, status_code=400)

    responseDict = {}
    try:
        uploaded_files = files

        for index, file in enumerate(uploaded_files):
            print(file.filename + " " + str(index))

        for index, file in enumerate(uploaded_files):
            filename = filenames[index]
            df_dataset = pd.DataFrame({})
            if index == 0:  # "Processing dataset table"
                df_dataset = await read_data(file)
                responseDict["datasetTable"] = df_dataset.head(15).to_json(orient='records')
                df_dataset.to_csv("dataset_table.csv", index=False)
            elif index == 1:  # "Processing index table"
                if df_dataset.empty:
                    df_dataset = pd.read_csv("dataset_table.csv")

                dataset_column_names = df_dataset.columns.tolist()


                label = {'geoCode': [], 'date': [], "time": []}
                geocode_keywords = ['zipcode', 'geography', "geo", "county", "fip"]
                date_keywords = ['date', 'time', 'datetime']
                time_keywords = ["time"]

                index_rows = []
                if file.filename.endswith('.csv'):
                    df_indexTable = await read_data(file)
                    column_names = df_indexTable.columns
                    index_rows = df_indexTable.values
                    for row in df_indexTable.values:
                        is_dataset_table = False
                        column_value = ""
                        for cell in row:
                            if cell in dataset_column_names:
                                is_dataset_table = True
                                column_value = cell
                            if is_dataset_table:
                                cell_content = str(cell).lower()
                                for keyword in geocode_keywords:
                                    if keyword in cell_content:
                                        label['geoCode'].append(column_value)

                                        break
                                for keyword in date_keywords:
                                    if keyword in cell_content:
                                        label['date'].append(column_value)
                                        break
                                for keyword in time_keywords:
                                    if keyword in cell_content:
                                        label['time'].append(column_value)
                                        break

                elif file.filename.endswith(('.xls', '.xlsx')):
                    workbook = await read_excel_two(file)
                    worksheet = workbook.active
                    for row in worksheet.iter_rows(values_only=True):


                        is_dataset_table = False
                        column_value = ""
                        for cell in row:
                            if cell in dataset_column_names:
                                is_dataset_table = True
                                column_value = cell
                            if is_dataset_table:
                                cell_content = str(cell).lower()
                                for keyword in geocode_keywords:
                                    if keyword in cell_content:
                                        label['geoCode'].append(column_value)
                                        break
                                for keyword in date_keywords:
                                    if keyword in cell_content:
                                        label['date'].append(column_value)
                                        break
                                for keyword in time_keywords:
                                    if keyword in cell_content:
                                        label['time'].append(column_value)
                                        break
                        if is_dataset_table:
                            index_rows.append(row)
                    if index_rows:
                        num_columns = len(index_rows[0])
                        column_names = [f'column{i + 1}' if i != 0 else 'variableName' for i in range(num_columns)]
                    else:
                        column_names = []

                df_index = pd.DataFrame(index_rows, columns=column_names)
                df_index.to_csv("index_table.csv", index=False)  
                responseDict["label"] = label
                responseDict["indexTable"] = df_index.head(15).to_json(orient='records')
            else:
                pass
        return JSONResponse(content=responseDict, status_code=200)
    except Exception as e:
        print(str(e))
        return JSONResponse(content={'error': str(e)}, status_code=500)

