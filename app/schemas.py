from typing import Union
from pydantic import BaseModel, Field


class DataRequest(BaseModel):
    overall_qual: int = Field(..., alias='Overall Qual')
    exter_qual: str = Field(..., alias='Exter Qual')
    bsmt_qual: str = Field(..., alias='Bsmt Qual')
    total_bsmt_sf: float = Field(..., alias='Total Bsmt SF')
    first_flr_sf: int = Field(..., alias='1st Flr SF')
    gr_liv_area: int = Field(..., alias='Gr Liv Area')
    kitchen_qual: str = Field(..., alias='Kitchen Qual')
    garage_cars: Union[float, int] = Field(..., alias='Garage Cars')
    garage_area: float = Field(..., alias='Garage Area')

    class Config:
        allow_population_by_field_name = True
