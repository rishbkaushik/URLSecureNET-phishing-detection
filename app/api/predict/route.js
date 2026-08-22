import { NextResponse } from "next/server";

export async function POST(request) {
  try {
    const formData = await request.formData();

    const response = await fetch(
      "http://urlsecurenet-backend:5000/predict",
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();

    return NextResponse.json(data, {
      status: response.status,
    });

  } catch (error) {
    console.error("Backend connection error:", error);

    return NextResponse.json(
      {
        error: "Unable to connect to Flask backend",
      },
      {
        status: 500,
      }
    );
  }
}