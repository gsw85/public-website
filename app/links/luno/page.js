import { FullPageLoading } from "@/components/loading";
import { redirect } from "next/navigation";

export default function Page() {
  redirect(process.env.LUNO);
  return <FullPageLoading />;
}
