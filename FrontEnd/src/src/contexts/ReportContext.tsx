// src/context/ReportsContext.tsx
import React, { createContext, useState, useContext } from "react";

export type Report = {
  id: number;
  clss: string;
  trust: number;
  title: string;
  description: string;
  treatment: string;
  image: string;
  created_at: string;
};

type ReportsContextType = {
  reports: Report[];
  setReports: React.Dispatch<React.SetStateAction<Report[]>>;
  addReport: (report: Report) => void;
};

const ReportsContext = createContext<ReportsContextType | undefined>(undefined);

export const ReportsProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [reports, setReports] = useState<Report[]>([]);

  const addReport = (newReport: Report, skipIfExists = false) => {
    setReports((prev) => {
      if (skipIfExists && prev.some((r) => r.id === newReport.id)) {
        return prev; // não adiciona duplicado
      }
      return [newReport, ...prev];
    });
  };
  return (
    <ReportsContext.Provider value={{ reports, setReports, addReport }}>
      {children}
    </ReportsContext.Provider>
  );
};

export const useReports = () => {
  const context = useContext(ReportsContext);
  if (!context)
    throw new Error("useReports must be used within ReportsProvider");
  return context;
};
