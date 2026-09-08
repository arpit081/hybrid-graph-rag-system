from pydantic import BaseModel


class EnterpriseQueryInput(BaseModel):
    text: str
    stream: bool = False


class HospitalQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
